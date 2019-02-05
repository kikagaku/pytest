"""
全てのテストケースが通っている場合にファイルを更新する
"""
from glob import glob
import subprocess

# 実行済みのファイルに更新する
def update_and_remove():
    # テストファイルの削除
    paths = glob('test_*.py')
    for path in paths:
        subprocess.call(['rm', path])
    # 実行ファイルの更新
    paths = glob('test_*.ipynb')
    for path in paths:
        name = path.split('test_')[1]
        subprocess.call(['mv', path, name])

if __name__ == '__main__':
    with open('result.log', mode='rb') as f:
        log = str(f.read())
        if 'FAILURES' not in log:
            print('ファイルの更新を行います。')
            update_and_remove()
        else:
            print('errorが残っています。')
