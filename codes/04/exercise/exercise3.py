import torch
from models import ExerciseModel

if __name__=="__main__":
    
    # 入力のテンソル定義
    in_tensor = torch.ones((32, 3, 128, 128))

    # モデルインスタンス作成
    model = ExerciseModel()

    # 実行 ＆ 結果確認
    out = model(in_tensor)
    print(repr(out.size()))
