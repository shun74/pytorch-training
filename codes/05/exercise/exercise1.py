from pathlib import Path

if __name__ == "__main__":
    
    data_directory = "./data"
    data_directory_path = Path(data_directory).resolve()
    print("===== problem1 =====")
    print(data_directory_path)

    dir_list = sorted(list(data_directory_path.glob("*")))
    print("===== problem2 =====")
    for dir in dir_list:
        print(dir)

    file_list = []
    for dir in dir_list:
        file_path_list = list(dir.glob("*"))
        file_list += file_path_list
    print("===== problem3 =====")
    print(len(file_list))