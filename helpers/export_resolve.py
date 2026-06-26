#!/usr/bin/env python3
"""
export_resolve.py — converte edl.json (formato video-use) em UM ÚNICO FCPXML
para importar no DaVinci Resolve, com os cortes já posicionados na timeline
como se a decupagem tivesse sido feita manualmente.

PASSO 0 (automático, sempre primeiro): converte cada source de .mp4 para
.mov com áudio PCM, requisito do DaVinci Resolve no Linux/Fedora (áudio AAC
dentro de .mp4 frequentemente não funciona no Resolve nessa plataforma).
O XML final referencia os .mov convertidos, nunca o .mp4 original.

Depois disso, tudo num arquivo só:
  - Cada `range` do EDL vira um clipe cortado (start/end) na timeline principal,
    na ordem em que aparecem — exatamente como um corte manual.
  - Cada clipe carrega um marker com beat/reason/quote, pra você navegar.
  - Color grade (preset ou "auto") é embutido como correção de cor estilo
    ASC CDL por clipe (Lift/Gamma/Gain + Saturação) — editável depois
    no Color Page do Resolve. CONFIABILIDADE: cortes e markers são robustos;
    a cor embutida é best-effort (ver nota no código). Se não vier, o nome
    do preset fica salvo no marker do clipe como rede de segurança.
  - Overlays (animações renderizadas) entram numa segunda trilha (lane),
    posicionados no tempo certo por cima do corte principal.
  - Avisos sobre loudnorm / HDR / legendas (que não entram no XML) ficam
    como marker de texto no primeiro clipe.

Uso:
    python helpers/export_resolve.py <edl.json> -o <saida.fcpxml>
    python helpers/export_resolve.py edit/edl.json -o edit/timeline.fcpxml --fps 30

Flags:
    --fps N                    Força um fps fixo em vez de detectar via ffprobe
    --no-auto-grade-analysis   Se grade="auto", não roda a análise signalstats
                                por clipe (mais rápido, mas sem CDL real)
    --skip-mov-conversion      NÃO recomendado: pula a conversão para .mov.
                                Use só se os sources já estiverem em .mov com
                                áudio compatível com o Resolve.

Requer ffmpeg/ffprobe no PATH.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------
# Presets de grade (copiados de helpers/grade.py para manter os MESMOS
# valores numéricos usados no pipeline ffmpeg, sem reinventar nada)
# ---------------------------------------------------------------------------

PRESETS = {
    "subtle":         {"contrast": 1.03, "brightness": 0.0,  "saturation": 0.98, "balance": None},
    "neutral_punch":  {"contrast": 1.06, "brightness": 0.0,  "saturation": 1.0,  "balance": None},
    "warm_cinematic": {
        "contrast": 1.12, "brightness": -0.02, "saturation": 0.88,
        # rs/gs/bs = shadows, rm/gm/bm = mids, rh/gh/bh = highlights (de grade.py)
        "balance": {
            "shadows":   (0.02, 0.0, -0.03),
            "midtones":  (0.04, 0.01, -0.02),
            "highlights": (0.08, 0.02, -0.05),
        },
    },
    "none": {"contrast": 1.0, "brightness": 0.0, "saturation": 1.0, "balance": None},
}


# ---------------------------------------------------------------------------
# Conversão preset -> ASC CDL (Slope / Offset / Power, por canal RGB)
# ---------------------------------------------------------------------------

def preset_to_cdl(preset_name: str):
    """Converte um preset nomeado de grade.py em valores ASC CDL aproximados.

    Mapeamento (aproximação deliberada, documentada):
      - eq.contrast  -> Slope (ganho geral em todos os canais) ~ Gain
      - eq.brightness -> Offset geral (somado ao offset por canal, se houver) ~ Lift
      - colorbalance shadows/mids/highlights -> Offset por canal
        (CDL não distingue zona tonal; usamos a média ponderada, mid >> resto)
      - eq.saturation -> Saturation (campo separado do CDL, fora de Slope/Offset/Power)
      - curves S-curve -> NÃO representável em CDL puro; ignorada (ASC CDL não
        tem curva paramétrica). Se isso importar visualmente, ajuste no Resolve
        com uma Curve node manual — fica anotado no marker do clipe.

    Power fica em 1.0 sempre (não há gamma explícito nos presets atuais).
    """
    p = PRESETS[preset_name]
    slope = [p["contrast"]] * 3
    offset = [p["brightness"]] * 3
    power = [1.0, 1.0, 1.0]
    sat = p["saturation"]

    if p["balance"]:
        # Soma o offset de cor (predominância midtone, que é onde a maior
        # parte da imagem cai) ao offset geral já calculado.
        r, g, b = p["balance"]["midtones"]
        offset = [offset[0] + r, offset[1] + g, offset[2] + b]

    return {"slope": slope, "offset": offset, "power": power, "saturation": sat}


# ---------------------------------------------------------------------------
# Auto-grade: mesma lógica matemática de grade.py::auto_grade_for_clip,
# mas devolvendo valores CDL em vez de string de filtro ffmpeg.
# ---------------------------------------------------------------------------

def _sample_frame_stats(video: Path, start: float, duration: float, n_samples: int = 10):
    fps = max(0.5, min(n_samples / max(duration, 0.1), 10.0))
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
        metadata_path = f.name
    try:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-nostats",
            "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{duration:.3f}",
            "-vf", f"fps={fps:.2f},signalstats,metadata=print:file={metadata_path}",
            "-f", "null", "-",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        y_avgs, y_mins, y_maxs, sat_avgs = [], [], [], []
        bit_depth = 8

        def parse(line):
            try:
                return float(line.rsplit("=", 1)[1])
            except (ValueError, IndexError):
                return None

        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if "lavfi.signalstats.YBITDEPTH" in line:
                    v = parse(line); bit_depth = int(v) if v else bit_depth
                elif "lavfi.signalstats.YAVG" in line:
                    v = parse(line); y_avgs.append(v) if v is not None else None
                elif "lavfi.signalstats.YMIN" in line:
                    v = parse(line); y_mins.append(v) if v is not None else None
                elif "lavfi.signalstats.YMAX" in line:
                    v = parse(line); y_maxs.append(v) if v is not None else None
                elif "lavfi.signalstats.SATAVG" in line:
                    v = parse(line); sat_avgs.append(v) if v is not None else None

        if not y_avgs:
            return {"y_mean": 0.5, "y_range": 0.7, "sat_mean": 0.25}

        max_val = (2 ** bit_depth) - 1
        y_mean = (sum(y_avgs) / len(y_avgs)) / max_val
        y_range = (
            ((sum(y_maxs) / len(y_maxs)) - (sum(y_mins) / len(y_mins))) / max_val
            if y_maxs and y_mins else 0.7
        )
        sat_mean = ((sum(sat_avgs) / len(sat_avgs)) / max_val) if sat_avgs else 0.25
        return {"y_mean": y_mean, "y_range": y_range, "sat_mean": sat_mean}
    except Exception:
        return {"y_mean": 0.5, "y_range": 0.7, "sat_mean": 0.25}
    finally:
        Path(metadata_path).unlink(missing_ok=True)


def auto_grade_to_cdl(video: Path, start: float, duration: float):
    """Réplica da lógica de decisão de grade.py::auto_grade_for_clip,
    devolvendo CDL (slope/offset/power/saturation) em vez de string ffmpeg."""
    stats = _sample_frame_stats(video, start, duration)
    y_mean, y_range, sat_mean = stats["y_mean"], stats["y_range"], stats["sat_mean"]

    contrast_adj = 1.08 - 0.05 * max(0.0, min(1.0, (y_range - 0.50) / 0.15)) if y_range < 0.65 else 1.03

    gamma_adj = 1.0
    if y_mean < 0.42:
        t = max(0.0, min(1.0, (y_mean - 0.30) / 0.12))
        gamma_adj = 1.10 - 0.08 * t
    elif y_mean > 0.60:
        gamma_adj = 0.97

    sat_adj = 0.98
    if sat_mean < 0.18:
        sat_adj = 1.04
    elif sat_mean > 0.38:
        sat_adj = 0.96

    contrast_adj = max(0.94, min(1.08, contrast_adj))
    gamma_adj = max(0.94, min(1.10, gamma_adj))
    sat_adj = max(0.94, min(1.06, sat_adj))

    return {
        "slope": [contrast_adj] * 3,
        "offset": [0.0, 0.0, 0.0],
        "power": [gamma_adj] * 3,  # gamma ~ power no modelo CDL
        "saturation": sat_adj,
    }


# ---------------------------------------------------------------------------
# Conversão mp4 -> mov com áudio PCM (REQUISITO CRÍTICO no Fedora/Linux)
# ---------------------------------------------------------------------------

def convert_to_mov(src_path: Path, out_dir: Path) -> Path:
    """Converte um arquivo de origem para .mov com áudio PCM, requisito do
    DaVinci Resolve no Linux/Fedora (AAC dentro de .mp4 frequentemente não
    reproduz áudio corretamente no Resolve nessa plataforma).

    Vídeo é copiado sem recodificar (rápido, sem perda). Áudio é convertido
    para PCM 16-bit 48kHz, que o Resolve sempre lê sem problema.

    Idempotente: se o .mov de saída já existe, não reconverte.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (src_path.stem + ".mov")

    if out_path.exists():
        print(f"  já convertido, pulando: {out_path.name}")
        return out_path

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_path),
        "-c:v", "copy",
        "-c:a", "pcm_s16le", "-ar", "48000",
        str(out_path),
    ]
    print(f"  convertendo para .mov (áudio PCM): {src_path.name} -> {out_path.name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Fallback: alguns perfis de vídeo não podem ser copiados direto para
        # .mov. Recodifica vídeo como H.264 mantendo qualidade alta.
        print(f"  copy direto falhou, recodificando vídeo (H.264 CRF 16): {src_path.name}")
        cmd_fallback = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(src_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-ar", "48000",
            str(out_path),
        ]
        subprocess.run(cmd_fallback, check=True)

    return out_path


def convert_all_sources(sources: dict, out_dir: Path) -> dict:
    """Converte todos os sources do EDL para .mov e retorna um novo dict
    {nome: caminho_mov} para uso no resto do export."""
    converted = {}
    for name, path in sources.items():
        src_path = Path(path)
        if not src_path.exists():
            print(f"  aviso: source '{name}' não encontrado em {src_path}, mantendo caminho original")
            converted[name] = path
            continue
        if src_path.suffix.lower() == ".mov":
            converted[name] = str(src_path)
            continue
        mov_path = convert_to_mov(src_path, out_dir)
        converted[name] = str(mov_path)
    return converted


# ---------------------------------------------------------------------------
# Utilidades de tempo / fps
# ---------------------------------------------------------------------------

def ffprobe_fps(path) -> Fraction:
    return ffprobe_info(path)["fps"]


def ffprobe_info(path) -> dict:
    """Retorna fps, width, height, duration, start_time e audio_channels do arquivo."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "0",
                "-show_entries",
                "stream=r_frame_rate,width,height,channels,codec_type"
                ":format=duration,start_time",
                "-of", "json", str(path),
            ],
            stderr=subprocess.DEVNULL,
        ).decode()
        data = json.loads(out)
    except Exception:
        return {
            "fps": Fraction(30, 1), "width": 1920, "height": 1080,
            "duration": 0.0, "start_time": 0.0, "audio_channels": 2,
        }

    fps = Fraction(30, 1)
    width, height = 1920, 1080
    audio_channels = 2

    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and "r_frame_rate" in s:
            try:
                num, den = s["r_frame_rate"].split("/")
                fps = Fraction(int(num), int(den))
            except Exception:
                pass
            width = s.get("width", width)
            height = s.get("height", height)
        elif s.get("codec_type") == "audio":
            audio_channels = s.get("channels", audio_channels)

    fmt = data.get("format", {})
    try:
        duration = float(fmt.get("duration", 0.0))
    except Exception:
        duration = 0.0
    try:
        start_time = float(fmt.get("start_time", 0.0))
    except Exception:
        start_time = 0.0

    return {
        "fps": fps, "width": width, "height": height,
        "duration": duration, "start_time": start_time,
        "audio_channels": audio_channels,
    }


def secs_to_rational(seconds: float, fps: Fraction) -> str:
    frame_count = round(seconds * fps.numerator / fps.denominator)
    num = frame_count * fps.denominator
    den = fps.numerator
    return f"{num}/{den}s"


# ---------------------------------------------------------------------------
# Construção do FCPXML
# ---------------------------------------------------------------------------

def cdl_to_filter_xml(cdl: dict, indent: str = "          ") -> str:
    """Gera um <filter-video> com parâmetros estilo ASC CDL (Lift/Gamma/Gain).

    AVISO DE CONFIABILIDADE: este bloco usa a convenção de nomes de parâmetro
    que o FCPXML usa para Color Correction básica. O importador de FCPXML do
    Resolve geralmente honra isso, mas não é garantido em todas as versões.
    Se a cor não vier visualmente no Resolve, os CORTES ainda vêm perfeitos —
    use o marker (com o nome do preset) para aplicar manualmente na Color Page.
    """
    s = cdl["slope"]
    o = cdl["offset"]
    pw = cdl["power"]
    sat = cdl["saturation"]
    lines = [
        f'{indent}<filter-video ref="cdl" name="Color Correction">',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Master|Saturation" value="{sat:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Shadows|Red" value="{o[0]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Shadows|Green" value="{o[1]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Shadows|Blue" value="{o[2]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Highlights|Red" value="{s[0]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Highlights|Green" value="{s[1]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Highlights|Blue" value="{s[2]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Midtones|Red" value="{pw[0]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Midtones|Green" value="{pw[1]:.4f}"/>',
        f'{indent}  <param name="Color Correction|Color Corrector 3-Way|Midtones|Blue" value="{pw[2]:.4f}"/>',
        f'{indent}</filter-video>',
    ]
    return "\n".join(lines) + "\n"


def build_fcpxml(edl: dict, edit_dir: Path, fps_override=None, compute_auto_grade=True) -> str:
    sources = edl["sources"]
    ranges = edl["ranges"]
    grade_field = edl.get("grade")
    overlays = edl.get("overlays") or []

    fps_by_source = {name: (fps_override or ffprobe_fps(path)) for name, path in sources.items()}
    seq_fps = next(iter(fps_by_source.values())) if fps_by_source else Fraction(30, 1)

    # Cache de info ffprobe por source (evita chamar N vezes o mesmo arquivo)
    source_info_cache = {name: ffprobe_info(path) for name, path in sources.items()}

    # ---- resources: 1 asset por CLIP (padrão que o DaVinci Resolve espera ao
    # importar FCPXML — ele próprio exporta um <asset> único por ocorrência na
    # timeline, mesmo que vários apontem para o mesmo arquivo fonte) ----
    resources = []
    next_id = 1

    # Formato da sequência (usa fps do primeiro source)
    if sources:
        first_name = next(iter(sources))
        first_fps = fps_by_source[first_name]
        first_info = source_info_cache[first_name]
        seq_fdur = f"{first_fps.denominator}/{first_fps.numerator}s"
        resources.append(
            f'    <format id="r_fmt_seq" width="{first_info.get("width", 1920)}" '
            f'height="{first_info.get("height", 1080)}" frameDuration="{seq_fdur}" />\n'
        )

    # Um asset por range/clip
    clip_asset_ids = []
    for r in ranges:
        src = r["source"]
        if src not in sources:
            raise ValueError(f"Range referencia source desconhecido '{src}'")
        path = sources[src]
        afps = fps_by_source[src]
        info = source_info_cache[src]
        fdur = f"{afps.denominator}/{afps.numerator}s"
        abs_path = os.path.abspath(path)
        audio_channels = info.get("audio_channels", 2)
        width = info.get("width", 1920)
        height = info.get("height", 1080)
        dur_str = secs_to_rational(info.get("duration", 0.0), afps)
        start_str_asset = secs_to_rational(info.get("start_time", 0.0), afps)
        asset_id = f"r{next_id}"; next_id += 1
        clip_asset_ids.append(asset_id)
        resources.append(
            f'    <format id="r_fmt_{asset_id}" width="{width}" height="{height}" frameDuration="{fdur}" />\n'
            f'    <asset id="{asset_id}" name="{escape(src)}" '
            f'hasVideo="1" hasAudio="1" audioSources="1" audioChannels="{audio_channels}" '
            f'format="r_fmt_{asset_id}" duration="{dur_str}" start="{start_str_asset}">\n'
            f'        <media-rep src="file://{escape(abs_path)}" kind="original-media"/>\n'
            f'    </asset>\n'
        )

    overlay_asset_ids = []
    for ov in overlays:
        asset_id = f"r{next_id}"; next_id += 1
        overlay_asset_ids.append(asset_id)
        ov_path = os.path.abspath(str((edit_dir / ov["file"]) if not Path(ov["file"]).is_absolute() else ov["file"]))
        ov_fps = ffprobe_fps(ov_path)
        fdur = f"{ov_fps.denominator}/{ov_fps.numerator}s"
        ov_info = ffprobe_info(ov_path)
        ov_dur_str = secs_to_rational(ov_info.get("duration", 0.0), ov_fps)
        resources.append(
            f'    <format id="r_fmt_{asset_id}" width="{ov_info.get("width", 1920)}" height="{ov_info.get("height", 1080)}" frameDuration="{fdur}" />\n'
            f'    <asset id="{asset_id}" name="{escape(Path(ov["file"]).name)}" '
            f'hasVideo="1" hasAudio="0" '
            f'format="r_fmt_{asset_id}" duration="{ov_dur_str}" start="0/1s">\n'
            f'        <media-rep src="file://{escape(ov_path)}" kind="original-media"/>\n'
            f'    </asset>\n'
        )

    # ---- clipes principais (cortes), em ordem = decupagem ----
    clips = []
    total_offset = 0.0
    warnings_note_added = False

    for i, r in enumerate(ranges):
        src = r["source"]
        start, end = float(r["start"]), float(r["end"])
        dur = end - start
        asset_id = clip_asset_ids[i]  # cada clip tem seu próprio asset ID único
        afps = fps_by_source[src]

        offset_str = secs_to_rational(total_offset, afps)
        duration_str = secs_to_rational(dur, afps)
        start_str = secs_to_rational(start, afps)
        one_frame_str = secs_to_rational(afps.denominator / afps.numerator, afps)
        clip_name = f"{src} ({r.get('beat', '')})".strip()

        note_parts = []
        if r.get("beat"):
            note_parts.append(f"Beat: {r['beat']}")
        if r.get("reason"):
            note_parts.append(f"Reason: {r['reason']}")
        if r.get("quote"):
            note_parts.append(f"Quote: {r['quote']}")

        if not warnings_note_added:
            warns = []
            if edl.get("subtitles"):
                warns.append(f"Legendas geradas em: {edl['subtitles']} (importar separado no Resolve)")
            warns.append("Áudio: aplicar loudness normalization (-14 LUFS / -1 dBTP / LRA 11) manualmente no Resolve, se desejado")
            warns.append("Se a fonte for HDR (HLG/PQ), verificar tone-mapping no Color Management do Resolve")
            note_parts.append("AVISOS: " + " | ".join(warns))
            warnings_note_added = True

        note = " || ".join(note_parts)

        grade_filter_xml = ""
        if grade_field and grade_field != "none":
            try:
                if grade_field == "auto" and compute_auto_grade:
                    cdl = auto_grade_to_cdl(Path(sources[src]), start, dur)
                elif grade_field in PRESETS:
                    cdl = preset_to_cdl(grade_field)
                else:
                    cdl = None
                if cdl:
                    grade_filter_xml = cdl_to_filter_xml(cdl)
            except Exception as e:
                note_parts.append(f"[grade export falhou: {e}]")

        # marker do beat — start="0s" é relativo ao início do asset-clip
        marker_xml = ""
        if r.get("beat"):
            marker_xml = (
                f'          <marker start="0s" duration="{one_frame_str}" '
                f'value="{escape(r["beat"])}" note="{escape(note)}"/>\n'
            )

        clips.append(
            f'        <asset-clip name="{escape(clip_name)}" ref="{asset_id}" '
            f'offset="{offset_str}" duration="{duration_str}" start="{start_str}" '
            f'format="r_fmt_{asset_id}" tcFormat="NDF">\n'
            f'{marker_xml}'
            f'{grade_filter_xml}'
            f'        </asset-clip>\n'
        )
        total_offset += dur

    # ---- overlays numa segunda trilha (lane="1"), por cima do corte principal ----
    overlay_clips = []
    for ov, asset_id in zip(overlays, overlay_asset_ids):
        ov_offset = float(ov["start_in_output"])
        ov_dur = float(ov["duration"])
        overlay_clips.append(
            f'        <asset-clip name="{escape(Path(ov["file"]).name)}" ref="{asset_id}" '
            f'lane="1" offset="{secs_to_rational(ov_offset, seq_fps)}" '
            f'duration="{secs_to_rational(ov_dur, seq_fps)}" '
            f'format="r_fmt_{asset_id}" tcFormat="NDF"/>\n'
        )

    total_duration_str = secs_to_rational(total_offset, seq_fps if not ranges else fps_by_source[ranges[-1]["source"]])

    fcpxml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.10">
  <resources>
{''.join(resources)}  </resources>
  <library>
    <event name="video-use export">
      <project name="video-use timeline">
        <sequence format="r_fmt_seq" duration="{total_duration_str}" tcStart="0s" tcFormat="NDF">
          <spine>
{''.join(clips)}{''.join(overlay_clips)}          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
'''
    return fcpxml


def main():
    ap = argparse.ArgumentParser(description="Exporta edl.json para UM FCPXML editável no DaVinci Resolve")
    ap.add_argument("edl_path", help="Caminho para o edl.json")
    ap.add_argument("-o", "--output", required=True, help="Caminho de saída do .fcpxml")
    ap.add_argument("--fps", type=float, default=None, help="Força um fps fixo")
    ap.add_argument("--no-auto-grade-analysis", action="store_true",
                     help="Se grade='auto', não roda signalstats (mais rápido, sem CDL real por clipe)")
    ap.add_argument("--skip-mov-conversion", action="store_true",
                     help="NÃO recomendado: pula a conversão mp4->mov com áudio PCM. "
                          "Use só se os sources já estiverem em .mov com áudio compatível.")
    args = ap.parse_args()

    edl_path = Path(args.edl_path)
    with open(edl_path) as f:
        edl = json.load(f)

    edit_dir = edl_path.parent
    if not args.skip_mov_conversion:
        mov_dir = edit_dir / "mov_for_resolve"
        print(f"Convertendo sources para .mov (áudio PCM) em {mov_dir} ...")
        edl["sources"] = convert_all_sources(edl["sources"], mov_dir)
        print("Conversão concluída.\n")
    else:
        print("Pulando conversão para .mov (--skip-mov-conversion). "
              "Atenção: áudio AAC em .mp4 pode não funcionar no Resolve/Linux.\n")

    fps_override = Fraction(args.fps).limit_denominator(1000) if args.fps else None
    xml = build_fcpxml(
        edl, edl_path.parent, fps_override=fps_override,
        compute_auto_grade=not args.no_auto_grade_analysis,
    )

    with open(args.output, "w") as f:
        f.write(xml)

    print(f"FCPXML escrito em {args.output} ({len(edl['ranges'])} clipes, {len(edl.get('overlays') or [])} overlays)")
    print("Importe no Resolve: File > Import > Timeline... (selecione este .fcpxml)")


if __name__ == "__main__":
    main()
