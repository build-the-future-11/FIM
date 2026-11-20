"""Render a checksum-bound A3 review poster from retained, published assets."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A3
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font-dir', type=Path, default=Path('/usr/share/fonts/truetype/dejavu'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    spec_path = here / 'poster_spec.json'
    spec = json.loads(spec_path.read_text())
    source_root = here.parent.resolve()
    for name, expected in spec['input_sha256'].items():
        source = (source_root / name).resolve()
        if source.parent != source_root or sha256(source) != expected:
            raise ValueError('Retained input identity differs: ' + name)
    if args.output.exists():
        raise FileExistsError('Choose a new output path; retained PDFs are not overwritten.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('PosterBody', str(args.font_dir / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('PosterBold', str(args.font_dir / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('PosterBody', normal='PosterBody', bold='PosterBold')
    ink, accent, muted = colors.HexColor('#173042'), colors.HexColor('#9B4438'), colors.HexColor('#536777')
    styles = {
        'title': ParagraphStyle('title', fontName='PosterBold', fontSize=28, leading=34, textColor=ink),
        'finding': ParagraphStyle('finding', fontName='PosterBold', fontSize=20, leading=26, textColor=accent),
        'heading': ParagraphStyle('heading', fontName='PosterBold', fontSize=15, leading=20, textColor=ink),
        'body': ParagraphStyle('body', fontName='PosterBody', fontSize=12, leading=17, textColor=ink),
        'caption': ParagraphStyle('caption', fontName='PosterBody', fontSize=10, leading=14, textColor=muted),
    }
    width, height = A3
    margin, gap = 40, 28
    content = width - margin * 2
    canvas = Canvas(str(args.output), pagesize=A3, pageCompression=1, invariant=1)
    canvas.setTitle(spec['title'])
    canvas.setAuthor('Author list pending review')
    canvas.setSubject('Retained-report presentation draft; no new scientific outcomes')

    def para(text, x, y, w, style='body', bottom=82):
        block = Paragraph(text, styles[style])
        _, h = block.wrap(w, height)
        if y - h < bottom:
            raise ValueError('Poster text exceeds its page region: ' + text[:60])
        block.drawOn(canvas, x, y - h)
        return y - h

    def figure(name, x, y, w):
        asset = ImageReader(str(source_root / name))
        original_w, original_h = asset.getSize()
        h = w * original_h / original_w
        canvas.drawImage(asset, x, y - h, width=w, height=h, mask='auto')
        return y - h

    y = height - margin
    y = para(escape(spec['kicker']), margin, y, content, 'caption') - 10
    y = para(escape(spec['title']), margin, y, content, 'title') - 13
    y = para('Presentation draft. Author list and affiliations require review.', margin, y, content, 'caption') - 16
    canvas.setStrokeColor(ink)
    canvas.setLineWidth(1)
    canvas.line(margin, y, width - margin, y)
    y -= 16
    y = para(escape(spec['finding']), margin, y, content, 'finding') - 15
    y = figure(spec['main_figure'], margin, y, content) - 7
    y = para(escape(spec['main_caption']), margin, y, content, 'caption') - 21
    col_width = (content - gap) / 2
    lowest = y
    for index, blocks in enumerate(spec['columns']):
        x, col_y = margin + index * (col_width + gap), y
        for block in blocks:
            if block['kind'] == 'heading':
                col_y = para(escape(block['text']), x, col_y, col_width, 'heading') - 8
            elif block['kind'] == 'text':
                col_y = para(escape(block['text']), x, col_y, col_width) - 12
            elif block['kind'] == 'image':
                col_y = figure(block['file'], x, col_y, col_width) - 8
            elif block['kind'] == 'caption':
                col_y = para(escape(block['text']), x, col_y, col_width, 'caption') - 14
            elif block['kind'] == 'table':
                data = [[Paragraph(escape(str(v)), styles['caption']) for v in row] for row in block['rows']]
                table = Table(data, colWidths=[col_width * f for f in block['widths']])
                table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LINEBELOW', (0, 0), (-1, 0), 0.8, ink),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('TOPPADDING', (0, 0), (-1, -1), 5),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ]))
                _, h = table.wrap(col_width, height)
                if col_y - h < 82:
                    raise ValueError('Poster table exceeds its page region.')
                table.drawOn(canvas, x, col_y - h)
                col_y -= h + 12
            else:
                raise ValueError('Unknown poster block.')
        lowest = min(lowest, col_y)
    if lowest < 82:
        raise ValueError('Poster content overlaps the provenance footer.')
    canvas.setStrokeColor(colors.HexColor('#C9D2D8'))
    canvas.line(margin, 72, width - margin, 72)
    para('Source: retained analysis and figures in the accompanying repository package.', margin, 62, content, 'caption', bottom=10)
    para('AI-assisted layout and wording. No new outcomes. A3 review layout; venue format and author review remain.', margin, 45, content, 'caption', bottom=10)
    canvas.linkURL(spec['source_url'], (margin, 48, width-margin, 64), relative=0)
    canvas.save()
    receipt = {
        'pdf': args.output.name, 'pdf_sha256': sha256(args.output),
        'layout': 'A3 portrait review draft; final venue dimensions not asserted',
        'pages': 1, 'input_sha256': spec['input_sha256'],
        'poster_spec_sha256': sha256(spec_path), 'builder_sha256': sha256(Path(__file__)),
        'python': platform.python_version(), 'reportlab': reportlab.Version,
        'scientific_outcome_regeneration': False, 'checkpoint_reads': 0,
        'original_artifacts_modified': False, 'submitted': False,
        'author_list_verified': False, 'source_url': spec['source_url'],
    }
    receipt_path = args.output.with_suffix('.receipt.json')
    if receipt_path.exists():
        raise FileExistsError('Receipt exists: ' + str(receipt_path))
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': receipt['pdf_sha256'], 'pages': 1}))


if __name__ == '__main__':
    main()
