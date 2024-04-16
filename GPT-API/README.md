# GPT-API

Here, we use the GPT-3.5-turbo API to generate summaries of the various transcribed pathological reports of patients. The API is used to generate summaries of the reports, which are then used to generate a final summary of the patient's condition. The final summary is constrained to a specific length, whcih is set by the user. The prompting is done using a chain of thought technique as mentioend in the PathLDM paper.

To run this code, you will need to have an OpenAI API key. You can get one by signing up at https://beta.openai.com/signup/. Once you have your API key, you can set it as an environment variable in your terminal by running:

```
export OPENAI_API_KEY="your-api-key"
```

You can then run the code by running:

```
./run.sh
```


Kindly note that it is necessary to have the following files in your input directory before running the run script
1. input_folder/summaries/summaries_list_train.json
2. input_folder/summaries/summaries_list_test.json
3. input_folder/TCGA_Reports.csv
4. input_folder/input.json

All these files can be found in the following google drive link: [DriveLink](https://drive.google.com/drive/folders/1j7XQXG-ZKibjhYvg-YRYh_z03sCr_bVM?usp=sharing)

This input path must be specified in the ```run.sh``` script to ensure that the correct files are being read. This section of the pipeline requires a stable internet connection to run as it uses the OpenAI API to generate summaries.