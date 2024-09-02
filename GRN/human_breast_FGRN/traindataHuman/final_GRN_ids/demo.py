import os
import pandas as pd

def load_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')

def compare_files(file_list):
    if len(file_list) < 2:
        print("至少需要两个文件进行对比")
        return
    
    first_df = load_tsv(file_list[0])
    for file in file_list[1:]:
        df = load_tsv(file)
        if not first_df.equals(df):
            print(f"文件 {file_list[0]} 和 {file} 不一样")
            return
    print("所有文件都相同")

def main(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tsv')]
    if len(files) != 4:
        print("文件夹中没有找到四个TSV文件，请检查文件夹内容")
        return
    
    compare_files(files)

# 调用主函数
folder_path = "./"  # 替换为你的文件夹路径
main(folder_path)
