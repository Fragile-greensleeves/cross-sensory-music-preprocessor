"""Export the reviewed analysis for downstream readers, without source media/text."""
import argparse
import hashlib
import json
from pathlib import Path

from musicprep.reference import package_digest
from musicprep.report import _public_package, write_report
from musicprep.validation import validate_package


def export_handoff(package, output, release=None):
    errors = validate_package(package)
    if errors:
        raise ValueError('; '.join(errors))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    public = _public_package(package)
    # Keep only a short human-readable review summary; full before/after evidence
    # stays in the private package. Never erase the original partial task status.
    review = package.get('melody', {}).get('provenance', {}).get('melody_review')
    if review:
        public['publication']['review_applied'] = True
    errors = validate_package(public)
    if errors:
        raise ValueError('; '.join(errors))
    data = json.dumps(public, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    (output / 'song-package.json').write_bytes(data)
    write_report(package, output / 'report.html', public=True)
    notes = public['melody']['notes']
    manifest = {
        'release': release or 'handoff-' + package_digest(package)[:16],
        'schema_version': public['schema_version'],
        'data_file': 'song-package.json',
        'data_sha256': hashlib.sha256(data).hexdigest(),
        'reviewed_source_package_digest': package_digest(package),
        'audio_source_sha256': public['audio']['source_sha256'],
        'duration_seconds': public['audio']['duration_seconds'],
        'counts': {'notes': len(notes), 'chords': len(public['harmony']['segments']),
                   'beats': len(public['rhythm']['beats']),
                   'downbeats': sum(b['is_downbeat'] is True for b in public['rhythm']['beats']),
                   'sections': len(public['sections']), 'descriptions': len(public['descriptions']),
                   'private_lyrics_lines': public['publication']['lyrics_line_count'],
                   'private_aligned_units': public['publication']['lyrics_word_count']},
        'note_sources': {s: sum(n['source'] == s for n in notes) for s in sorted({n['source'] for n in notes})},
        'excluded': ['source audio', 'separated stems', 'lyrics text and semantic excerpts',
                     'raw model output', 'local paths', 'model weights'],
        'status': public['run']['status'],
        'validation_errors': [],
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', help='Explicit version label for this handoff')
    args = parser.parse_args()
    if args.output.resolve() == args.package.parent.resolve():
        parser.error('Use a separate output directory; never overwrite the full package.')
    manifest = export_handoff(json.loads(args.package.read_text()), args.output, args.release)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
