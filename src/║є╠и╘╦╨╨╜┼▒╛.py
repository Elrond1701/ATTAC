import subprocess

def run_shell_script(script_path):
    # 执行shell脚本，并将输出重定向到PIPE
    process = subprocess.Popen(['sh', script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # 读取并打印脚本的输出
    while True:
        output = process.stdout.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # 等待脚本执行完毕
    process.wait()

# 测试
script_path = '/hy-tmp/continual_learning_with_vit/src/test_ewc.sh'
run_shell_script(script_path)