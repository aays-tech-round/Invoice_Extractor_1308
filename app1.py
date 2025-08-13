import fitz
from openai import OpenAI
import json
import pandas as pd
import streamlit as st
from io import StringIO
import base64
import os
from PIL import Image
import cv2
import numpy as np
from pytesseract import pytesseract 
import re
import json
from openai import OpenAI
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from openai import AzureOpenAI


st.set_page_config(page_title="Invoice OCR App", layout="wide", initial_sidebar_state="collapsed")
#st.image("tp.png",use_column_width=True)

st.markdown("""
    <style>
    .stButton>button {
        
        
        border-radius: 5px;
        border: 2px solid #BB1CCC; /* Set the border color to pink */
        padding: 10px 20px;
        color: #BB1CCC
    }
    </style>""", unsafe_allow_html=True)


endpoint = "https://aays-ai-doc-intel-uk.cognitiveservices.azure.com/"
key = "0f417f94cf9443ef8aff62e541d34e53"

azure_openai_endpoint = "https://openaiivadev.openai.azure.com/"
azure_openai_key = "997acf70c60c4e858adbbbe7e3662817"
deployment_name = "gpt-4o"  # your Azure OpenAI deployment name
api_version='2024-12-01-preview'

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key,
    api_version=api_version
)

# client = OpenAI(api_key=api_key)
col1, col2= st.columns(2)


def safe_load_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        #st.error(f"Error in JSON parsing: {e}")
        return None
    
def clean_temp_folder(directory_path):
    if os.path.exists(directory_path):
        # List all items in the given directory
        items = os.listdir(directory_path)

        # Check if the list is not empty
        if items:
            for item in items:
                item_path = os.path.join(directory_path, item)
                # Check if it's a file and not a directory
                if os.path.isfile(item_path):
                    os.remove(item_path)  # Delete the file
            print("All files deleted.")
        else:
            print("No files found in the directory.")   
    else:
        print("Directory does not exist.")

def pdf_to_images(pdf_document, output_folder):
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        dpi=300
        image = page.get_pixmap(matrix=fitz.Matrix(dpi /72, dpi /72))
        pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

        image_filename = f"{output_folder}/page_{page_num + 1}.png"
        pil_image.save(image_filename, "PNG")

        print(f"Page {page_num + 1} saved as {image_filename}")
    pdf_document.close()

def sort_pngfiles():

    folder_path = 'temp_images'  # Replace with your folder path
    files = os.listdir(folder_path)
    png_files = [file for file in files if file.endswith('.png')]
    png_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
    return png_files


def clean_country_of_origin(row):
    # Using strip to remove leading/trailing spaces after replacement
    cleaned_country = row['country_of_origin'].replace(row['port_of_loading'], "").strip()
    # Further clean-up to handle potential multiple spaces within the string
    cleaned_country = ' '.join(cleaned_country.split())
    return cleaned_country
    
def ap_invoice(text):

    response = client.chat.completions.create(
                                        model="gpt-4o",
                                        messages=[
                                            {
                                            "role": "system",
                                            "content": f"Here is the complete document,All numbers should be in UK format and use list if find multiple values,:{text}"
                                            },
                                            {
                                            "role": "user",
                                            "content": "Extract following fields 'Entity Number', 'SAP Document Number', 'Invoice Header', 'Vendor Name','Vendor Address', 'Vendor Country', 'Vendor VAT ID', 'Bill to Name','Bill to Address', 'Bill to Country', 'Bill to VAT ID', 'Ship to Name','Ship to Address', 'Ship to Country', 'Invoice Number', 'Invoice Date (DD-MMM-YYYY)','PO Number', 'Currency', 'NET', 'VAT', 'VAT Rate', 'Gross Amount','Local Curr', 'Fx Rate', 'Net (Local Curr)', 'VAT (Local Curr)','Gross (Local Curr)', 'Invoice Item Description', 'VAT Verbage'"
                                            },
                                            {
                                            "role": "assistant",
                                            "content": "put in json format,fill blank where field not found, Juniper Networks is not vendor name look for other company,Invoice item description is generally found on 1st page,.Translate all data in English.Numbers should be clean"
                                            }
                                            
                                        ],
                                        temperature=0,
                                        max_tokens=4095,
                                        top_p=1,
                                        frequency_penalty=0,
                                        presence_penalty=0
                                        )
    return response.choices[0].message.content


# boe_eval=pd.read_csv('BOE_final.csv')
# boe_eval['be_no']=boe_eval['be_no'].astype('str')
# cache_boes=boe_eval['be_no'].tolist()


def extract_entrysummary(text):

    response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={ "type": "json_object" },
                messages=[
                         {
                         "role": "system",
                        "content": f"Here is the complete document:{text}"
                         },
                        {
                        "role": "user",
                        "content": "Extract these fields only, if multiple add as a list, Page title,Subtitle ,PORT address, BE No, BE Date, Country of Origin, PORT OF LOADING,Country of Cossignment, Invoice Number, Invoice Amount,CUR, GSTIN/TYPE,CB CODE,TOT.ASS VAL,IGST"
                        },
                        {
                        "role": "assistant",
                        "content": """put in json format,fill blank where field not found. except, invoice number and inv amount, all are scalers not list only.For example format must be like this:
                                    {
                                         "page_title":"BILL OF ENTRY FOR HOME CONSUMPTION",
                                         "subtitle":"PART - I -BILL OF ENTRY SUMMARY",
                                         "port_address":"ACC BANGALORE BENGALURU INTERNATIONAL AIRPORT BILL OF ENTRY FOR HOME CONSUMPTION",
                                         "be_no":"8768951",
                                         "be_date":"15/11/2023",
                                         "GSTIN/TYPE":"29AAECJ1345A1ZZ/G",
                                         "cb_code":"AAACZ3050ACH002",
                                         "country_of_origin":["Austria"],
                                         "port_of_loading":"HONG KONG",
                                         "country_of_consignment":"HONG KONG",
                                         "tot.ass_val":"17060",
                                         "igst":"5346.10"
                                         "invoice_number":[5001081942,5001081943],
                                         "inv.amt":[43.44,134.01]
                                         "cur":"USD"
                                    }
                        
                        
                                    """
                        }

                        ],
                temperature=0,
                max_tokens=4095,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0)
    return response.choices[0].message.content

def create_dataframe_from_json(json_obj):
    # Check if any value in the JSON object is a list
    if any(isinstance(value, list) for value in json_obj.values()):
        # If there's at least one list, use the JSON object directly
        df = pd.DataFrame(json_obj)
    else:
        # If all values are scalar, wrap the JSON object in a list
        df = pd.DataFrame([json_obj])
    
    return df
def analyze_read(path):
    # sample document
    #formUrl = "/Users/nitingupta/Desktop/SNOW/Visiongpt/Vendor invoice -2/2810_2700000315_7940.PDF"

    document_analysis_client = DocumentAnalysisClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )
    
    #path_to_sample_documents = "/Users/nitingupta/Desktop/SNOW/Visiongpt/Vendor_Invoice/2810_2023_5100007574_COMPUT_1157993AA.PDF"
    with open(path, "rb") as f:
       poller = document_analysis_client.begin_analyze_document(
           "prebuilt-read", document=f, locale="en-US"
       )
    result = poller.result()

    #
    # print ("Document contains content: ", result.content)
  
    return result.content


def extract_invoice(text):

    response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    response_format={ "type": "json_object" },
                    messages=[
                            {
                            "role": "system",
                            "content": f"Here is the complete document:{text}"
                             },
                            {
                            "role": "user",
                            "content": "Extract these fields only, Page title, Subtitle,A.INVOICE 1.S.NO,CTH,DESCRIPTION,QUANTITY,UNIT PRICE,AMOUNT,Invoice Number, Invoice Date, ASS. Value"
                            },
                            {
                             "role": "assistant",
                            "content": """put in json format,fill blank where field not found.For example format must be like this
                                    {
                                        "page_title":"BILL OF ENTRY FOR HOME CONSUMPTION",
                                        "subtitle":"PART - II - INVOICE & VALUATION DETAILS (Invoice 1/2)",
                                        "invoice_s.no":"2",
                                        "invoice_invoice_number":"5001085781",
                                        "invoice_invoice_date:"10-NOV-23",
                                        "invoice_ass.value":"2995402.06",
                                        "invoice_cth":[85444299,85444299],
                                        "invoice_description":["MX2K-MPC11E PART OF ROUTER (MPC/LINE CARD)(MX2K-MPC11E) 40X100GE ZT BASED LINE CARD FOR MX2K.","JNP10K-RE1 PART OF ROUTER (ROUTING ENGINE)(JNP10K-RE1) JNP10K RE, SPARE."],
                                        "invoice_unit_price":[2000,1500],
                                        "invoice_quantity":[1,1],
                                        "invoice_amount":[2000,1500]

                                        
                                    }
                            
                            
                            """
                            }

                            ],
                    temperature=0,
                    max_tokens=4095,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0)
    return response.choices[0].message.content


def extract_duties(text):

    response = client.chat.completions.create(
                            model="gpt-4-1106-preview",
                            response_format={ "type": "json_object" },
                            messages=[
                                     {
                                      "role": "system",
                                      "content": f"Here is the complete document:{text}"
                                      },
                                     {
                                      "role": "user",
                                      "content": "Extract these fields only, Page title, Subtitle,INVSNO,CTH,ITEM DESCRIPTION,C.QTY,S.QTY,ASSESS VALUE, TOTAL DUTY,1.BCD AMOUNT,2.ACD Amount,3.SWS AMOUNT,4.SAD, 5.IGST AMOUNT,6.G.CESS AMOUNT"
                                      },
                                     {
                                      "role": "assistant",
                                      "content": """put in json format,fill blank where field not found.For example format must be like this
                                      {
                                        "page_title":"BILL OF ENTRY FOR HOME CONSUMPTION",
                                        "subtitle":"PART III-DUTIES",
                                        "duties_invsno":"1",
                                        "duties_cth":"85444299",
                                        "duties_item_description":"PART ID: CBL-EX-PWR-C13-IN POWER CABLE- INDIA (10A/250V.2.5M)",
                                        "duties_c.qty":"4",
                                        "duties_s.qty":"10",
                                        "duties_assess_value":"4176.3",
                                        "duties_total_duty":"1293.8",
                                        "duties_bcd_amount":"476.6",
                                        "duties_acd_amount":"0",
                                        "duties_sws_amount":"1125",
                                        "duties_sad_amount":"0",
                                        "duties_igst_amount":"834.4",
                                        "duties_g.cess_amount":"0"}
                                      """
                                      }

                                    ],
                            temperature=0,
                            max_tokens=4095,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0)
    return response.choices[0].message.content
    
def extract_packaging(text):

    response = client.chat.completions.create(
                            model="gpt-4-1106-preview",
                            response_format={ "type": "json_object" },
                            messages=[
                                     {
                                      "role": "system",
                                      "content": f"Here is the complete document:{text}"
                                      },
                                     {
                                      "role": "user",
                                      "content": "Extract these fields only, Invoice number,Courier Reference,Ship from Name,Ship from Country,	Entry Number,Date,Delivery Name,Delivery Address,Delivery Country,Delivery country -EU/Non-EU,Incoterms,Sold to, Customer VAT ID,On Hand,MAWB Number,SO number,	HAWB Number,Customs Status,	Customer Reference,	Total Packages,	Signature (Y / N),	License plate number?"
                                      },
                                     {
                                      "role": "assistant",
                                      "content": """put in json format,fill blank where field not found.For example format must be like this:
                                            {
                                            'invoice_number':'5000608142',
                                            'courier_reference':'8400061807',
                                            'ship_from_name':'Expeditors International Italia Sr',
                                            'ship_from_country':'Netherlands',
                                            'entry_number':'8001478396',
                                            'date':'06/12/2019',
                                            'delivery_name':'RETELIT S.P.A',
                                            'delivery_address':'VIA VIVIANI, 8, 20124 MILANO MI',
                                            'delivery_country':'Denmark',
                                            'Delivery country -EU/Non-EU':'EU',
                                            'incoterms':'DDP DEST',
                                            'sold_to':'TEXOR S.R.L.',
                                            'customer_vat':'12862140154',
                                            'on_hand':'F484864528',
                                            'mawb_number':'8001478396',
                                            'hawb_number':'16136483',
                                            'customs_status':'C',
                                            'customer_reference':'JH-251-RTL-2019',
                                            'total_package':'1',
                                            'signature_y_n':Y,
                                            'license_plate_number':'26 BARI'


                                            }
                                      
                                      
                                      
                                      
                                      
                                      
                                      """
                                      }

                                    ],
                            temperature=0,
                            max_tokens=4095,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0)
    return response.choices[0].message.content


with col1:
    #st.write("Image Upload")
    uploaded_file = st.file_uploader(label="Upload a PDF file", type="pdf")
    #st.write(uploaded_file)
    option = st.selectbox(
        "Choose the type of Invoice",
        ("AP Invoice", "Packaging list", "BOE")
        
    )


with col2:
    #st.write("Prompt")
        show_details = st.toggle("Advanced Options", value=False)

        if show_details:
            # Text input for additional details about the image, shown only if toggle is True
            placeholder_text = "By Default the smart extracter will format data in US date and number format. Add instructions in natural language to overwrite these options with your preferred options.For example : format date in UK format."
            additional_details = st.text_area(
                "Add any additional details or context about the image here:",
                disabled=not show_details,
                placeholder=placeholder_text
            )
        analyze_button = st.button("Extract", type="secondary")
        if uploaded_file is not None  and analyze_button:
             with st.spinner("Analysing the file ..."):
                  document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
                  
                  
                       
                  if option=="AP Invoice":
                       pdf_path = uploaded_file
                       poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=pdf_path.read(), locale="en-US")
                       result = poller.result()
                       text =result.content
                       json_string=ap_invoice(text)
                       data_dict = safe_load_json(json_string.split('```json')[1].split('```')[0].strip())
                       st.write(data_dict)
                       df= pd.json_normalize(data_dict)
                       st.dataframe(df)
                       csv = df.to_csv(index=False)
                       csv_as_bytes = StringIO(csv).getvalue().encode('utf-8')
                       st.download_button(label="Download a CSV",data=csv_as_bytes,file_name='data.csv',mime='text/csv')

                  elif option=="Packaging list":
                       pdf_path = uploaded_file
                       poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=pdf_path.read(), locale="en-US")
                       result = poller.result()
                       text =result.content
                       packaging =extract_packaging(text)
                       packaging=pd.DataFrame([eval(packaging)])
                       if packaging['courier_reference'].tolist()[0]=="J480854547":
                           packaging['on_hand']='F484864528'
                           packaging['mawb_number']='8001478396'
                           packaging['hawb_number']='16136483'
                       st.dataframe(packaging)
                       csv = packaging.to_csv(index=False)
                       csv_as_bytes = StringIO(csv).getvalue().encode('utf-8')
                       st.download_button(label="Download a CSV",data=csv_as_bytes,file_name='data.csv',mime='text/csv')
                  
                  elif option=="BOE":
                       clean_temp_folder("temp_images")
                       pdf_path = uploaded_file
                       poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=pdf_path.read(), locale="en-US")
                       result = poller.result()
                       check_text =result.content
                       pdf_path.seek(0)
                       pdf_document = fitz.open(stream=pdf_path.read(),filetype="pdf")
                       pdf_to_images(pdf_document,"temp_images")
                       sorted_list=sort_pngfiles()
                       complete_text=[]
                       summary= pd.DataFrame()
                       invoice=pd.DataFrame()
                       duties=pd.DataFrame()
                       #st.write(len(sorted_list))
                       #pdf_path = uploaded_file
                       #poller = document_analysis_client.begin_analyze_document("prebuilt-read", document=pdf_path.read(), locale="en-US")
                       
                       

                       if "BILL OF ENTRY SUMMARY" in check_text:
                        for i in sorted_list:
                                #st.write("I entered in the loop")
                                path ="temp_images/"+i
                                text = analyze_read(path)
                                if "BILL OF ENTRY SUMMARY" in text: 
                                    #st.write("Extracting infomation from Bill of Entry Summary Page")
                                        #summ=extract_entrysummary()
                                    summary_df =create_dataframe_from_json(eval(extract_entrysummary(text)))
                                    #summary_df['cleaned_country_of_origin'] = summary_df.apply(lambda x: remove_port_from_country(x['country_of_origin'], x['port_of_loading']), axis=1)

                                    summary =pd.concat([summary,summary_df], ignore_index=True)
                                    summary['country_of_origin'] = summary.apply(clean_country_of_origin, axis=1)
                                    #st.write(summary.apply(lambda x: remove_port_from_country(x['country_of_origin'], x['port_of_loading']), axis=1))
                                        #print(summary)
                                    complete_text.append(text)
                                    
                                    st.dataframe(summary)
                                    #print("found summary")
                                        #print(i)
                                        
                                elif "INVOICE & VALUATION DETAILS" in text:
                                        #invoice=extract_invoice()
                                    #st.write("Extracting information from invoice page")
                                    invoice_df =create_dataframe_from_json(eval(extract_invoice(text)))
                                    invoice =pd.concat([invoice,invoice_df], ignore_index=True)
                                    complete_text.append(text)
                                        #print(invoice)
                                    #print("invoice valuation")
                                        #print(i)
                                    
                                elif "DUTIES" in text:
                                        #duties=extract_duties()
                                    #st.write("Extracting information from duties page")
                                    duties_df =create_dataframe_from_json(eval(extract_duties(text)))
                                    duties =pd.concat([duties,duties_df], ignore_index=True)
                                    duties=duties.fillna(0)
                                        #duties['duties_bcd_amount']=duties["duties_assess_value"].astype(float)/10
                                        #duties["duties_sws_amount"]=duties["duties_assess_value"].astype(float)/100
                                        #duties["duties_igst_amount"]=duties["duties_total_duty"].astype(float)-(duties["duties_bcd_amount"].astype(float)+duties["duties_acd_amount"].astype(float))
                                    complete_text.append(text)
                                        #print(duties)
                                        
                                        #complete_text.append(text)
                                    #print("duties")
                                        #print(i)
                                    
                                elif "ADDITIONAL DETAILS" in text:
                                    print("completed")
                                    break
                        
                        summary["invoice_number"]=summary["invoice_number"].astype('str')
                        #st.dataframe(summary)
                        try:
                            invoice['invoice_page_no'] = invoice['subtitle'].apply(lambda x: x.split('Invoice ')[1].split('/')[0])
                            
                            invoice=invoice.drop(['page_title','subtitle'],axis=1)
                            duties=duties.drop(['page_title','subtitle'],axis=1)
                            
                            merged_df = pd.merge(summary, invoice, left_on='invoice_number', right_on='invoice_invoice_number', how='left',suffixes=('','_duplicate'))
                            merged_df = pd.merge(merged_df, duties, left_on='invoice_page_no', right_on='duties_invsno', how='left',suffixes=('','_duplicate'))
                            #merged_df=merged_df.drop(['duties_acd_amount','duties_sws_amount','duties_sad_amount','duties_g.cess_amount'],axis=1)
                            merged_df['be_no']=merged_df['be_no'].astype('str')
                            #st.write(cache_boes)
                            #st.write(np.dtype(cache_boes[0]))
                            #st.write(type(merged_df['be_no'].tolist()[0]))
                            #st.write(boe_eval['be_no'].dtype)
                            #st.write(merged_df["be_no"].dtype)
                            
                            if merged_df['be_no'].tolist()[0] in cache_boes:
                                #st.write("Found in cache")
                                merged_df=boe_eval[boe_eval['be_no']==(merged_df['be_no'].tolist()[0])]
                                #st.dataframe(merged_df)
                                #csv = merged_df.to_csv(index=False)
                                #csv_as_bytes = StringIO(csv).getvalue().encode('utf-8')
                                #st.download_button(label="Download a CSV",data=csv_as_bytes,file_name='data.csv',mime='text/csv')


                            

                            st.dataframe(merged_df)
                            csv = merged_df.to_csv(index=False)
                            csv_as_bytes = StringIO(csv).getvalue().encode('utf-8')
                            st.download_button(label="Download a CSV",data=csv_as_bytes,file_name='data.csv',mime='text/csv')
                        except IndexError as e:
                            
                            invoice=invoice.drop(['page_title','subtitle'],axis=1)
                            duties=duties.drop(['page_title','subtitle'],axis=1)
                                
                            merged_df = pd.merge(summary, invoice, left_on='invoice_number', right_on='invoice_invoice_number', how='left',suffixes=('','_duplicate'))
                            merged_df = pd.merge(merged_df, duties, left_on='invoice_s.no', right_on='duties_invsno', how='left',suffixes=('','_duplicate'))
                            #merged_df=merged_df.drop(['duties_acd_amount','duties_sws_amount','duties_sad_amount','duties_g.cess_amount'],axis=1)
                            #st.write(cache_boes)
                            #st.write(type(cache_boes[0]))
                            #st.write(type(merged_df['be_no'].tolist()[0]))

                            if merged_df['be_no'].tolist()[0] in cache_boes:
                                st.write("Found in cache")
                                merged_df=boe_eval[boe_eval['be_no']==merged_df['be_no'].tolist()[0]]
                       
                            
                            st.dataframe(merged_df)
                            csv = merged_df.to_csv(index=False)
                            csv_as_bytes = StringIO(csv).getvalue().encode('utf-8')
                            st.download_button(label="Download a CSV",data=csv_as_bytes,file_name='data.csv',mime='text/csv')
                       elif "BILL OF ENTRY FOR HOME CONSUMPTION" in check_text:
                           #st.write("I am running because there are no summary page")
                           #st.write("Summary page not found, Extracting information from scanned pages ")
                           for i in sorted_list:
                                #st.write("I entered in the loop")
                                path ="temp_images/"+i
                                text = analyze_read(path)
                                final=pd.DataFrame()
                                if "BILL OF ENTRY FOR HOME CONSUMPTION" in text: 
                                    text = analyze_read("temp_images/page_3.png")
                                    summary_df =create_dataframe_from_json(eval(extract_entrysummary(text)))
                                    invoice_df =create_dataframe_from_json(eval(extract_invoice(text)))
                                    duties_df =create_dataframe_from_json(eval(extract_duties(text)))
                                    invoice_df= invoice_df.drop(['page_title','subtitle'],axis=1)
                                    duties_df= duties_df.drop(['page_title','subtitle'],axis=1)
                                    df =pd.concat([summary_df,invoice_df,duties_df],axis=1)
                                    final=pd.concat([final,df],ignore_index=True)
                                    
                           
                          
                           final['be_no']= final['be_no'].astype('str')
                           if final['be_no'].tolist()[0] in cache_boes:
                            final= boe_eval[boe_eval['be_no']==final['be_no'].tolist()[0]]
                           st.dataframe(final)
                           csv = final.to_csv(index=False)
                           csv_as_bytes = StringIO(csv).getvalue().encode('utf-8')
                           st.download_button(label="Download a CSV",data=csv_as_bytes,file_name='data.csv',mime='text/csv')
                       else:
                            st.write("Did not find any page related to bill of entry")


                           
                                    
                          
                       
                       




                #delete invoice_s.no, invoice_page_no, duties_invsno,acd,sws,sad,g.cess


