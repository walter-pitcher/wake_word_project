from pathlib import Path
data = Path('data')

neg_dir = data / 'negative'
print('=== NEGATIVE FOLDER ===')
total_neg = 0
for sub in sorted(neg_dir.iterdir()):
    if sub.is_dir():
        count = len(list(sub.rglob('*.wav')))
        print(f'  {sub.name}: {count}')
        total_neg += count
print(f'Total in negative/: {total_neg}')

hard_top = data / 'hard_negatives_top'
hard_top_count = len(list(hard_top.rglob('*.wav'))) if hard_top.exists() else 0
print(f'\nhard_negatives_top/: {hard_top_count}')

for name in ['negative_manifest.txt', 'negative_manifest_hard.txt', 'negative_manifest_hard_merged.txt']:
    p = data / name
    if p.exists():
        print(f'{name}: {len(p.read_text().splitlines())} lines')
    else:
        print(f'{name}: NOT FOUND')