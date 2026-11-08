import py_compile
py_compile.compile('rng(ai).py', doraise=True)
print('COMPILE OK')

with open('rng(ai).py') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')
print(f'First line: {lines[0].strip()[:60]}...')
print(f'Last line: {lines[-1].strip()[:80]}')

with open('rng(ai).py') as f:
    content = f.read()

for key in ['FRAGMENT_SHADER', 'WorldMapWindow', 'run_gpu', 'main', 'fbm_point', 'biome_label', 'apply_sea_level_point', 'apply_ice_caps_point']:
    found = key in content
    print(f'{key}: {"FOUND" if found else "MISSING"}')