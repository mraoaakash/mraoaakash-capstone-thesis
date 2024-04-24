import os
import numpy as np
import pandas as pd
from openai import OpenAI

def get_encoding_k_summary(MESSAGE, TOKEN_LEN):
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception('API key is missing')
    first_part = f"Extract information from breast pathology report. List the histological classification, i.e. type of cancer or DCIS, subtype, description of any necrosis, any mention of tumor infiltrating lymphocytes, histological grade, nuclear grade, lymphovascular invasion, calcification, receptor status, IHC and any other ancillary testing results. List out and expand the main points. The report is: \n {MESSAGE}"
    second_message = f"Please generate a succinct report in {TOKEN_LEN} words from the above information. Focus more on the sample itself and less on the process to generate the sample. Exclude any filler words or sentences. If something is not mentioned or specified, exclude it from the report. Be as crisp as possible, and only include very important information. Extremely Low verbosity"

    structured = [
        {
            "role": "user",
            "content": first_part
        },
        {
            "role": "user",
            "content": second_message
        }
    ]

    return structured

def get_encoding_keywords(MESSAGE, TOKEN_LEN):
    if 'OPENAI_API_KEY' not in os.environ:
        raise Exception('API key is missing')
    first_part = f"Extract information from breast pathology report. List the histological classification, i.e. type of cancer or DCIS, subtype, description of any necrosis, any mention of tumor infiltrating lymphocytes, histological grade, nuclear grade, lymphovascular invasion, calcification, receptor status, IHC and any other ancillary testing results. List out the main keywords that are present in the report: \n {MESSAGE}"
    second_message = f"Please generate a succinct list of summary words from the above information.  Exclude any filler words or sentences. If something is not mentioned or specified, exclude it from the report. Be as crisp as possible, and only include very important information. Give only a list. Extremely Low verbosity"

    structured = [
        {
            "role": "user",
            "content": first_part
        },
        {
            "role": "user",
            "content": second_message
        }
    ]

    return structured



def get_response(MESSAGE, TOKEN_LEN, type='k_summary'):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=MESSAGE,
        stream=True,
        max_tokens=TOKEN_LEN if type == 'k_summary' else 1000,
    )
    output = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            output += chunk.choices[0].delta.content

    return output





# message = "Specimen #: (Age: )F Race: WHITE. FAMENDED. SPECIMEN: A: WIDE LOCAL EXCISION RIGHT BREAST B: SENT LN #1. C: SENT LN #2 D: SENT LN #1 (. E: SENT LN #2 (. F: NONSENT LN #3. FINAL DIAGNOSIS: A. BREAST, RIGHT, WIDE LOCAL EXCISION: - INVASIVE DUCTAL CARCINOMA, MODERATELY DIFFERENTIATED. (BLOOM-RICHARDSON: TUBULAR FORMATION-2, NUCLEI- - MITOSES-2 = 6) ,. 1. CM, NEGATIVE FOR LYMPHVASCULAR INVASION, MARGINS NEGATIVE. - FIBROCYSTIC CHANGES TO INCLUDE CYSTS, FIBROSIS, APOCRINE METAPLASIA,. AND DUCT ECTASIA. - MULTIFOCAL FAT NECROSIS. - ER/PR IMMUNOSTAINS ARE STRONGLY POSITIVE. - HER2-NEU OVEREXPRESSION BY FISH: NOT AMPLIFIED (1.2). B. LYMPH NODE, NON-SENTINAL #1, BIOPSY: TWO LYMPH NODES NEGATIVE FOR MALIGNANCY (0/2) . C. LYMPH NODE, NON-SENTINAL #2, BIOPSY: TWO LYMPH NODES NEGATIVE FOR MALIGNANCY (0/2) . D. LYMPH NODE, SENTINAL #1. BIOPSY. NEGATIVE FOR MALIGNANCY BY IMMUNOHISTOCHEMICAL STAINS. E. LYMPH NODE, SENTINAL #2. BIOPSY: NEGATIVE FOR MALIGNANCY BY IMMUNOHISTOCHEMICAL STAINS. F. LYMPH NODE, NON-SENTINAL #3, BIOPSY: BENIGN ADIPOSE TISSUE, NO LYMPHOID TISSUE PRESENT. PATHOLOGIC STAGING : Tlc NO MX. COMMENT: CYTOKERATIN IS NEGATIVE IN SPECIMEN D AND E. Specimen #: FINAL DIAGNOSIS (continued) : CLINICAL DIAGNOSIS AND HISTORY: -year-old white female with 2.5 cm right outer mid breast tumor. Fine. needle aspiration positive for atypical cells consistent with carcinoma. FROZEN SECTION DIAGNOSIS: A: RIGHT BREAST - INFILTRATING DUCTAL CARCINOMA. Results relayed to Dr. in person. GROSS DESCRIPTION: A. WIDE LOCAL EXCISION RIGHT BREAST received fresh and consists of a piece. of fatty tissue measuring 9.0 X 8.0 x 3.0 cm. The specimen is oriented. with sutures by surgeon. Ink code: Red=medial lateral, blue=superior,. green=inferior, yellow=anterior, black=posterior. Sectioning reveals a. firm, tan, well-circumscribed mass, 1.7 cm in greatest dimension in the. mid portion of the specimen; the mass is located 0. 8 cm from the deep. margin. A frozen section is prepared (block A1). There. is. a. second. well-circumscribed, tan nodule, 0.7 cm in diameter, located 0.5 cm. superior anterior to the main mass. Sections harvested for breast. protocol include (1) mass, OCT embedded (2) mass, flash frozen, and. (3) grossly normal breast from the medial end of the specimen, flash. frozen (matching paraffin sections=A1 and A2, respectively) . Additional. representative sections are submitted in sequential order from lateral to. medial in cassettes A3 through A12. First mass in cassettes A1, A5, A6,. A8, and the second mass in A7, A9, A10, and All. 12CF. B. SENTINEL LYMPH NODE NUMBER ONE 'NON-SENTINEL NODULE NUMBER ONE'. received in formalin and consists of two irregular fragments of tan soft. tissue with attached yellow adipose tissue measuring 2.0 and 1.7 cm in. greatest dimension. The fragments are bisected and submitted in their. entirety in two cassettes, one node per cassette. 2C4. C. SENTINEL LYMPH NODE NUMBER TWO NON-SENTINEL NODE NUMBER TWO' received. in formalin and consists of two irregular fragments of yellow, lobular. adipose tissue measuring 2.5 cm in greatest dimension each. The. fragments. are bisected and submitted in their entirety in two cassettes, one. fragment per cassette. 2C4. Specimen # : GROSS DESCRIPTION (continued). D. SENTINEL LYMPH NODE NUMBER ONE. received in formalin and consists. of one ovoid fragment of tan, fibrous tissue with attached yellow adipose. tissue measuring 2.5 cm in greatest dimension. The specimen is serially. sectioned and submitted in its entirety in two cassettes. 2C4. E. SENTINEL LYMPH NODE NUMBER TWO. received in formalin and consists. of a fragment of tan soft tissue measuring 2.5 cm in greatest dimension. The specimen is trisected and submitted in its entirety in two cassettes. 2C3. F. NON-SENTINEL LYMPH NODE NUMBER THREE received in formalin and consists. of one ovoid fragment of tan soft tissue with adherent yellow adipose. tissue measuring 1.7 cm in greatest dimension. The specimen is bisected. and submitted in its entirety in one cassette. 1C2:"
# message = get_encoding(message, 75)

# print(get_response(MESSAGE = message, TOKEN_LEN = 75))

# # The specimen was diagnosed as an invasive ductal carcinoma, moderately differentiated, with fibrocystic changes, multifocal fat necrosis, and ER/PR strongly positive. HER2-Neu not amplified and no lymphovascular invasion was present. Additionally, two nonsentinel lymph nodes and two sentinel lymph nodes were negative for malignancy, and another nonsentinel lymph node contained benign adipose tissue. Pathologic staging was T1c N0 MX. There was no mention of tumor infiltrating lymphocytes or calcification.
# # The patient has invasive ductal carcinoma in the right breast, moderately differentiated (Bloom-Richardson grade: 6). No lymphovascular invasion noted and margins are negative. Multifocal fat necrosis and fibrocystic changes were found. ER/PR immunostains were strongly positive and HER-2 neu overexpression was not amplified. All lymph nodes