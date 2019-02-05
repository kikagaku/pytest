from glob import glob
import subprocess

# jupyter notebookの実行
# 実行後のファイルは test_ としている
def run_notebook(path):
    cmd = 'jupyter nbconvert --to notebook --execute {} --output test_{} --allow-errors'.format(path, path).split()
    subprocess.check_call(cmd)

# テストファイルを削除する
def remove_tests():
    paths = glob('test_*.ipynb') + glob('test_*.py')
    for path in paths:
        subprocess.call(['rm', path])

# テストケース
def generate_script(path):
    script = """from glob import glob
import nbformat
import subprocess
import nbformat

def _extract_errors(path):
    nb = nbformat.read(path, nbformat.current_nbformat)
    errors = []
    for cell in nb.cells:
        for output in cell['outputs']:
            if output['output_type'] == 'error':
                errors.append(output)
    return errors

def test_ipynb():
    errors = _extract_errors('test_{}')
    assert errors == []
    """.format(path)
    return script

# テストファイルを生成する
if __name__ == '__main__':
    remove_tests()
    paths = glob('*.ipynb')
    for path in paths:
        run_notebook(path)
        script = generate_script(path)
        name = path.split('.ipynb')[0]
        with open('test_'+name+'.py', mode='w') as f:
            f.write(script)
