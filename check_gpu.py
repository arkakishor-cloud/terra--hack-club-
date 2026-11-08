import py_compile
py_compile.compile('rng(ai).py', doraise=True)
print('COMPILE OK')
with open('rng(ai).py') as f:
    content = f.read()
for key in ['WorldMapWindow=None', 'run_gpu=None', 'class WorldMapWindow', 'def run_gpu', 'FRAGMENT_SHADER']:
    print(f'{key}: {key in content}')
