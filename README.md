# 논문 제목

## 제3회 ETRI 휴먼이해 인공지능 논문경진대회

### 본 대회는 한국전자통신연구원(ETRI)이 주최하고 과학기술정보통신부와 국가과학기술연구회(NST)가 후원합니다.

> ⚙ 개발환경

![Python](https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white)
![Windows 10](https://img.shields.io/badge/Windows-0078D6.svg?&style=for-the-badge&logo=Windows&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-007ACC.svg?&style=for-the-badge&logo=Visual%20Studio%20Code&logoColor=white)

<br/>

> 활용 데이터
+ train: 2023년 데이터(105일치)
+ test: 2023년 데이터(115일치)
  + heart rate, mobile accelerator, mobile gps data
<br />

## Dependencies
Torch 2.0.1 and above.
<br /> **Installing Pytorch with CUDA support is recommended.**
<br /><br />


## Installation
Python 3.8.16
<br/>
OS type: Windows

```
pip install -r requirements.txt
```

## Pretrained Model

<table style="margin: auto">
  <thead>
    <tr>
      <th>Model</th>
      <th>Baseline<br />Name</th>
      <th>Hidden<br />Size</th>
      <th>Layer<br />Number</th>
      <th>Time<br />Shift</th>
      <th>Noise</th>
      <th>Class Imbalance
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Model 1 </td>
      <td align="right">LSTM</td>
      <td align="center">32</td>
      <td align="center">2</td>
      <td align="center">:x:</td>
      <td align="center">:x:</td>
      <td align="center">:white_check_mark:</td>
    </tr>
    <tr>
      <td>Model 2 </td>
      <td align="right">LSTM</td>
      <td align="center">32</td>
      <td align="center">2</td>
      <td align="center">:x:</td>
      <td align="center">:white_check_mark:</td>
      <td align="center">:white_check_mark:</td>
    </tr>
    <tr>
      <td>Model 3 </td>
      <td align="right">LSTM</td>
      <td align="center">32</td>
      <td align="center">2</td>
      <td align="center">:white_check_mark:</td>
      <td align="center">:white_check_mark:</td>
      <td align="center">:white_check_mark:</td>
    </tr>
    <tr>
      <td>Model 4 </td>
      <td align="right">LSTM</td>
      <td align="center">32</td>
      <td align="center">2</td>
      <td align="center">:x:</td>
      <td align="center">:white_check_mark:</td>
      <td align="center">:white_check_mark:</td>
    </tr>
    <tr>
      <td>Model 5 </td>
      <td align="right">LSTM</td>
      <td align="center">64</td>
      <td align="center">2</td>
      <td align="center">:x:</td>
      <td align="center">:x:</td>
      <td align="center">:x:</td>
    </tr>
  </tbody>
</table>

🌠**Model weight (.pth)**: [download](https://drive.google.com/file/d/1YUk-eAsYNSzoP0xFDqZ9xGNwvw1Yq_hp/view?usp=sharing)

<br />
가중치 파일은 <code>weights</code> 폴더 내부에 저장

```
ETRI_lifelog
   └──weights
      └──combined_model.pth
```

## Data Structure
### 🗂️ Raw Data
레포지토리 하위 폴더로 원본 데이터(raw data) 저장 및 <mark><b>이름 수정 필요</b></mark>
<br /><kbd>휴먼이해2024 ➡️ human2024</kbd>
<br /><kbd>val dataset ➡️ val_dataset</kbd>
<br /><kbd>test dataset ➡️ test_dataset</kbd>
<br />
```
ETRI_lifelog
   └──human2024
      ├── val_dataset
      │    ├── ch2024_val__m_acc_part_1.parquet.gzip
      │    ├── ch2024_val__m_acc_part_2.parquet.gzip
      │    ├── ...
      │    └──ch2024_val__w_pedo.parquet.gzip
      │ 
      └── test_dataset
           ├── ch2024_test__m_acc_part_5.parquet.gzip
           ├── ch2024_test__m_acc_part_6.parquet.gzip
           ├── ...
           └── ch2024_test_w_pedo.parquet.gzip
```

### 🗂️ Feature Data
학습 및 테스트 시, 원본 데이터(raw data)로부터 feature data가 생성되어 저장됨. (약 50MB)
```
ETRI_lifelog
   └──feature_data
      ├── train_ts
      │    └── merged
      │         ├── merged_1.csv
      │         ├── merged_2.csv
      │         ├── merged_3.csv
      │         └── merged_4.csv
      └── test_ts
           └── merged
                ├── merged_5.csv
                ├── merged_6.csv
                ├── merged_7.csv
                └── merged_8.csv
```
## ✨ Cross Validation
```
python cross_val.py \
      --train_data_root YOUR/ETRI2024/VAL/DATA/FOLDER/PATH \
      --label_path YOUR/VAL/LABEL/PATH/val_label.csv \
```

## ✨ Train
```
python trainer.py \
      --train_data_root YOUR/ETRI2024/VAL/DATA/FOLDER/PATH \
      --label_path YOUR/VAL/LABEL/PATH/val_label.csv \
```

## ✨ Test
```
python tester.py \
      --test_data_path YOUR/ETRI2024/TEST/DATA/FOLDER/PATH \
      -w YOUR/WEIGHT/PATH/combined_model2.pth \
```


