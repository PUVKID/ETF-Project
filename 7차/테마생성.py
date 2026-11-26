"""실제 Rust 테마 컴파일러 출력을 사용자 화면에 연결합니다."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
FRONT=ROOT/'web'
SHARED=Path('E:/2026/Astryx 공통 디자인/1차')
EXECUTABLE=SHARED/'target/release/astryx-theme.exe'
OFFICIAL=SHARED/'node_modules/@astryxdesign/theme-neutral/dist/theme.css'

def main():
    palettes=json.loads((ROOT/'테마설정.json').read_text(encoding='utf-8'))
    outputs=[]
    calls=[]
    for mode in ['light','dark']:
        argv=[str(EXECUTABLE),'--format','css','--mode',mode]
        for key,color in zip(['accent','background','surface','text','muted','border'],palettes[mode]):
            argv.extend(['--'+key,color])
        result=subprocess.run(argv,cwd=ROOT,capture_output=True,check=True,timeout=10)
        text=result.stdout.decode('utf-8',errors='strict')
        if '--color-primary:' not in text or '\ufffd' in text:
            raise ValueError('테마 출력 형식을 확인할 수 없습니다')
        outputs.append(text)
        calls.append({'argv':argv,'cwd':str(ROOT),'exit_code':result.returncode,'stdout_bytes':len(result.stdout),'sha256':hashlib.sha256(result.stdout).hexdigest()})
    raw='\n'.join(outputs).encode('utf-8')
    (FRONT/'astryx-theme.css').write_bytes(raw)
    (FRONT/'astryx-official.css').write_bytes(OFFICIAL.read_bytes())
    (ROOT/'work/Rust테마_실제호출_비공개.json').write_text(json.dumps({'calls':calls,'css_sha256':hashlib.sha256(raw).hexdigest(),'mode':'빌드 시 실제 호출','performance':'동일 조건 전후 성능 미측정'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Rust 실제 호출2회, 라이트·다크 CSS 연결:',FRONT)

if __name__=='__main__':main()
