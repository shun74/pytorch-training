from pathlib import Path

def get_files_in_directory(data_directory):
    assert (data_directory.exists() and data_directory.is_dir()),  "'data' directory not found."

    file_list = list(data_directory.glob("*"))

    return file_list

# ファイルが存在するか判定
def check_file_existence(file_path):
    path = Path(file_path)
    if path.exists():
        print(f"The file '{file_path}' exists.")
    else:
        print(f"The file '{file_path}' does not exist.")


if __name__ == "__main__":
    
    # 絶対パスの取得
    data_directory = "./data"
    data_directory_path = Path(data_directory).resolve()
    print("---Displaying the absolute path---")
    print(data_directory_path)

    # 配下のファイル取得
    files_in_data = get_files_in_directory(data_directory_path)
    print("---Display all files underneath---")
    for file in files_in_data:
        print(file)

    # pathの連結
    base_directory = Path("/base_dir")
    subdirectory_name = "data_dir"

    new_path = base_directory / subdirectory_name
    print("---Display the concatenated path---")
    print(new_path)    

    # ディレクトリを作成したいパス
    # directory_path = Path("./target_dir")
    # directory_path.mkdir(parents=True, exist_ok=True)