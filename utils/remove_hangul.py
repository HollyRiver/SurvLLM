import datasets
import re

def remove_hangul(sample, column):
    hangul_pattern = r"[\uAC00-\uD7A3]"

    cleaned_messages = []
    for message in sample[column]:
        # content 값에서 한글을 제거
        cleaned_content = re.sub(hangul_pattern, "", message["content"])
        
        # 정제된 content로 메시지 딕셔너리 재생성
        cleaned_messages.append({
            "role": message["role"],
            "content": cleaned_content
        })
    
    # 정제된 messages 리스트를 샘플에 다시 할당
    sample[column] = cleaned_messages
    return sample