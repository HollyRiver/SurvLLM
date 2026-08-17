import pandas as pd
import glob
import os


def excel_integrate(input_name):
    ## 엑셀 파일들이 있는 폴더 경로
    input_folder_path = f"logs/{input_name}/"

    ## 결과물로 저장할 엑셀 파일 경로
    output_file_path = f"logs/{input_name}/{input_name}.xlsx"

    ## 폴더 내의 모든 .csv 파일을 찾기
    file_list = glob.glob(input_folder_path + "*.csv")

    ## 하나의 Excel 파일(Writer)에 시트를 추가하며 쓰기
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        ## 찾/은 파일 리스트를 순회
        for file_path in file_list:
            df = pd.read_csv(file_path)
            
            ## 시트 이름 만들기 (파일 경로에서 파일명만 추출)
            base_name = os.path.basename(file_path)
            sheet_name = os.path.splitext(base_name)[0]
            
            ## ExcelWriter에 'sheet_name'이라는 시트로 df 저장
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"총 {len(file_list)}개의 파일이 '{output_file_path}' 파일 하나로 합쳐졌습니다.")