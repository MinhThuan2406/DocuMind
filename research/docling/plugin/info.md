[Repository](https://github.com/opendatalab/OmniDocBench) | [Paper](https://arxiv.org/abs/2412.07626) | [Dataset](https://huggingface.co/datasets/opendatalab/OmniDocBench)


## Docling Plugins

| Component | Default | OmniDocBench |
| --- | --- | --- |
| OCR option | `OcrAutoOptions` | `PP-OCRv6-medium` |
| Layout model | `docling-layout-heron` | `DocLayout-YOLO release-v2` |
| Table-structure model | `TableFormer` | `RapidTable SLANet-plus` |

### Hướng tiếp cận đánh giá tuần tự

Đánh giá và lựa chọn từng thành phần theo thứ tự phụ thuộc:

```text
Layout → OCR → Table Structure
```

Ở mỗi bước, chỉ thay đổi một thành phần và giữ lại model có kết quả tốt hơn làm baseline cho bước tiếp theo.

## Table of Contents

- [Text Recognition](#text-recognition)
- [Table Structure Recognition](#table-structure-recognition)
- [Layout Detection](#layout-detection)
- [Evaluated Models and Sources](#evaluated-models-and-sources)

## Text Recognition

OmniDocBench contains bounding box information and corresponding text recognition annotations for all text in each PDF page, making it suitable as a benchmark for OCR evaluation. The text annotations include both block-level and span-level annotations, both of which can be used for evaluation. This repo currently provides an example of block-level evaluation, which evaluates OCR at the text paragraph level.

<table style="width: 90%; margin: auto; border-collapse: collapse; font-size: small;">
  <thead>
    <tr>
      <th rowspan="2">Model Type</th>
      <th rowspan="2">Model</th>
      <th colspan="3">Language</th>
      <th colspan="3">Text background</th>
      <th colspan="4">Text Rotate</th>
    </tr>
    <tr>
      <th><i>EN</i></th>
      <th><i>ZH</i></th>
      <th><i>Mixed</i></th>
      <th><i>White</i></th>
      <th><i>Single</i></th>
      <th><i>Multi</i></th>
      <th><i>Normal</i></th>
      <th><i>Rotate90</i></th>
      <th><i>Rotate270</i></th>
      <th><i>Horizontal</i></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7" style="text-align: center;">Pipeline Tools<br>&<br>Expert Vision<br>Models</td>
      <td>PaddleOCR</td>
      <td>0.071</td>
      <td><b>0.055</b></td>
      <td><ins>0.118</ins></td>
      <td><b>0.060</b></td>
      <td><b>0.038</b></td>
      <td><ins>0.085</ins></td>
      <td><b>0.060</b></td>
      <td><b>0.015</b></td>
      <td><ins>0.285</ins></td>
      <td><b>0.021</b></td>
    </tr>
    <tr>
      <td>OpenOCR</td>
      <td>0.07</td>
      <td><ins>0.068</ins></td>
      <td><b>0.106</b></td>
      <td><ins>0.069</ins></td>
      <td>0.058</td>
      <td><b>0.081</b></td>
      <td><ins>0.069</ins></td>
      <td><ins>0.038</ins></td>
      <td>0.891</td>
      <td><ins>0.025</ins></td>
    </tr>
    <tr>
      <td>Tesseract-OCR</td>
      <td>0.096</td>
      <td>0.551</td>
      <td>0.250</td>
      <td>0.439</td>
      <td>0.328</td>
      <td>0.331</td>
      <td>0.426</td>
      <td>0.117</td>
      <td>0.969</td>
      <td>0.984</td>
    </tr>
    <tr>
      <td>EasyOCR</td>
      <td>0.26</td>
      <td>0.398</td>
      <td>0.445</td>
      <td>0.366</td>
      <td>0.287</td>
      <td>0.388</td>
      <td>0.36</td>
      <td>0.97</td>
      <td>0.997</td>
      <td>0.926</td>
    </tr>
    <tr>
      <td>Surya</td>
      <td>0.057</td>
      <td>0.123</td>
      <td>0.164</td>
      <td>0.093</td>
      <td>0.186</td>
      <td>0.235</td>
      <td>0.104</td>
      <td>0.634</td>
      <td>0.767</td>
      <td>0.255</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td><ins>0.033</ins></td>
      <td>0.240</td>
      <td>0.261</td>
      <td>0.185</td>
      <td>0.121</td>
      <td>0.166</td>
      <td>0.180</td>
      <td><ins>0.038</ins></td>
      <td><b>0.185</b></td>
      <td>0.638</td>
    </tr>
    <tr>
      <td>GOT-OCR</td>
      <td>0.041</td>
      <td>0.112</td>
      <td>0.135</td>
      <td>0.092</td>
      <td><ins>0.052</ins></td>
      <td>0.155</td>
      <td>0.091</td>
      <td>0.562</td>
      <td>0.966</td>
      <td>0.097</td>
    </tr>
    <tr>
      <td rowspan="3" style="text-align: center;">Vision Language<br>Models</td>
      <td>Qwen2-VL-72B</td>
      <td>0.072</td>
      <td>0.274</td>
      <td>0.286</td>
      <td>0.234</td>
      <td>0.155</td>
      <td>0.148</td>
      <td>0.223</td>
      <td>0.273</td>
      <td>0.721</td>
      <td>0.067</td>
    </tr>
    <tr>
      <td>InternVL2-76B</td>
      <td>0.074</td>
      <td>0.155</td>
      <td>0.242</td>
      <td>0.113</td>
      <td>0.352</td>
      <td>0.269</td>
      <td>0.132</td>
      <td>0.610</td>
      <td>0.907</td>
      <td>0.595</td>
    </tr>
    <tr>
      <td>GPT4o</td>
      <td><b>0.020</b></td>
      <td>0.224</td>
      <td>0.125</td>
      <td>0.167</td>
      <td>0.140</td>
      <td>0.220</td>
      <td>0.168</td>
      <td>0.115</td>
      <td>0.718</td>
      <td>0.132</td>
    </tr>
  </tbody>
</table>
<p>Component-level OCR text recognition evaluation on OmniDocBench (v1.0) text subset.</p>

OCR text recognition evaluation can be configured according to [ocr](https://github.com/opendatalab/OmniDocBench/blob/v1_5/configs/ocr.yaml). 

<details>
  <summary>Field explanation: ocr.yaml</summary>

The configuration file for `ocr.yaml` is as follows:

```YAML
recogition_eval:      # Specify task name, common for all recognition-related tasks
  metrics:            # Configure metrics to use
    - Edit_dist       # Normalized Edit Distance
    - BLEU
    - METEOR
  dataset:                                                                   # Dataset configuration
    dataset_name: omnidocbench_single_module_dataset                         # Dataset name, no need to modify if following the specified input format
    ground_truth:                                                            # Ground truth dataset configuration
      data_path: ./demo_data/recognition/OmniDocBench_demo_text_ocr.json     # JSON file containing both ground truth and model prediction results
      data_key: text                                                         # Field name storing Ground Truth, for OmniDocBench, text recognition results are stored in the text field, all block level annotations containing text field will participate in evaluation
    prediction:                                                              # Model prediction configuration
      data_key: pred                                                         # Field name storing model prediction results, this is user-defined
    category_type: text                                                      # category_type is mainly used for selecting data preprocessing strategy, options: formula/text
```

For the `dataset` section, the input `ground_truth` `data_path` follows the same data format as OmniDocBench, with a custom field under each text sample for storing model predictions. The prediction field is selected through `prediction.data_key`, for example `pred`. See [OmniDocBench_demo_text_ocr](https://github.com/opendatalab/OmniDocBench/blob/v1_5/demo_data/recognition/OmniDocBench_demo_text_ocr.json) for the expected input format.

Here is a reference model inference script for your consideration:

```PYTHON
import os
import json
from PIL import Image

def poly2bbox(poly):
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    bbox = [L, U, R, D]
    return bbox

question = "<image>\nPlease convert this cropped image directly into latex."

with open('./demo_data/omnidocbench_demo/OmniDocBench_demo.json', 'r') as f:
    samples = json.load(f)
    
for sample in samples:
    img_name = os.path.basename(sample['page_info']['image_path'])
    img_path = os.path.join('./Docparse/images', img_name)
    img = Image.open(img_path)

    if not os.path.exists(img_path):
        print('No exist: ', img_name)
        continue

    for i, anno in enumerate(sample['layout_dets']):
        if not anno.get('text'):             # Filter out annotations containing the text field from OmniDocBench for model inference
            continue

        bbox = poly2bbox(anno['poly'])
        im = img.crop(bbox).convert('RGB')
        response = model.chat(im, question)  # Modify the way the image is passed in according to the model
        anno['pred'] = response              # Directly add a new field to store the model's inference results under the corresponding annotation

with open('./demo_data/recognition/OmniDocBench_demo_text_ocr.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False)
```

</details>


## Table Structure Recognition

OmniDocBench contains bounding box information for tables on each PDF page along with corresponding table recognition annotations, making it suitable as a benchmark for table recognition evaluation. The table annotations are available in both HTML and LaTeX formats, with this repository currently providing examples for HTML format evaluation.

<table style="width: 100%; margin: auto; border-collapse: collapse; font-size: small;">
  <thead>
    <tr>
      <th rowspan="2">Model Type</th>
      <th rowspan="2">Model</th>
      <th colspan="3">Language</th>
      <th colspan="4">Table Frame Type</th>
      <th colspan="4">Special Situation</th>
      <th rowspan="2">Overall</th>
    </tr>
    <tr>
      <th><i>EN</i></th>
      <th><i>ZH</i></th>
      <th><i>Mixed</i></th>
      <th><i>Full</i></th>
      <th><i>Omission</i></th>
      <th><i>Three</i></th>
      <th><i>Zero</i></th>
      <th><i>Merge Cell</i>(+/-)</th>
      <th><i>Formula</i>(+/-)</th>
      <th><i>Colorful</i>(+/-)</th>
      <th><i>Rotate</i>(+/-)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2" style="text-align: center;">OCR-based Models</td>
      <td>PaddleOCR</td>
      <td><ins>76.8</ins></td>
      <td>71.8</td>
      <td>80.1</td>
      <td>67.9</td>
      <td>74.3</td>
      <td><ins>81.1</ins></td>
      <td>74.5</td>
      <td><ins>70.6/75.2</ins></td>
      <td><ins>71.3/74.1</ins></td>
      <td><ins>72.7/74.0</ins></td>
      <td>23.3/74.6</td>
      <td>73.6</td>
    </tr>
    <tr>
      <td>RapidTable</td>
      <td><b>80.0</b></td>
      <td><b>83.2</b></td>
      <td><b>91.2</b></td>
      <td><b>83.0</b></td>
      <td><b>79.7</b></td>
      <td><b>83.4</b></td>
      <td>78.4</td>
      <td><b>77.1/85.4</b></td>
      <td><b>76.7/83.9</b></td>
      <td><b>77.6/84.9</b></td>
      <td><ins>25.2/83.7</ins></td>
      <td><b>82.5</b></td>
    </tr>
    <tr>
      <td rowspan="2" style="text-align: center;">Expert VLMs</td>
      <td>StructEqTable</td>
      <td>72.8</td>
      <td><ins>75.9</ins></td>
      <td>83.4</td>
      <td>72.9</td>
      <td><ins>76.2</ins></td>
      <td>76.9</td>
      <td><b>88</b></td>
      <td>64.5/81</td>
      <td>69.2/76.6</td>
      <td>72.8/76.4</td>
      <td><b>30.5/76.2</b></td>
      <td><ins>75.8</ins></td>
    </tr>
    <tr>
      <td>GOT-OCR</td>
      <td>72.2</td>
      <td>75.5</td>
      <td><ins>85.4</ins></td>
      <td><ins>73.1</ins></td>
      <td>72.7</td>
      <td>78.2</td>
      <td>75.7</td>
      <td>65.0/80.2</td>
      <td>64.3/77.3</td>
      <td>70.8/76.9</td>
      <td>8.5/76.3</td>
      <td>74.9</td>
    </tr>
    <tr>
      <td rowspan="2" style="text-align: center;">General VLMs</td>
      <td>Qwen2-VL-7B</td>
      <td>70.2</td>
      <td>70.7</td>
      <td>82.4</td>
      <td>70.2</td>
      <td>62.8</td>
      <td>74.5</td>
      <td><ins>80.3</ins></td>
      <td>60.8/76.5</td>
      <td>63.8/72.6</td>
      <td>71.4/70.8</td>
      <td>20.0/72.1</td>
      <td>71.0</td>
    </tr>
    <tr>
      <td>InternVL2-8B</td>
      <td>70.9</td>
      <td>71.5</td>
      <td>77.4</td>
      <td>69.5</td>
      <td>69.2</td>
      <td>74.8</td>
      <td>75.8</td>
      <td>58.7/78.4</td>
      <td>62.4/73.6</td>
      <td>68.2/73.1</td>
      <td>20.4/72.6</td>
      <td>71.5</td>
    </tr>
  </tbody>
</table>
<p>Component-level Table Recognition evaluation on OmniDocBench(v1.0) table subset. <i>(+/-)</i> means <i>with/without</i> special situation.</p>


Table recognition evaluation can be configured according to [table_recognition](https://github.com/opendatalab/OmniDocBench/blob/v1_5/configs/table_recognition.yaml). 

**For tables predicted to be in LaTeX format, the [latexml](https://math.nist.gov/~BMiller/LaTeXML/) tool will be used to convert LaTeX to HTML before evaluation. The evaluation code will automatically perform format conversion, and users need to preinstall [latexml](https://math.nist.gov/~BMiller/LaTeXML/)**

<details>
  <summary>Field explanation: table_recognition.yaml</summary>

The configuration file for `table_recognition.yaml` is as follows:

```YAML
recogition_eval:      # Specify task name, common for all recognition-related tasks
  metrics:            # Configure metrics to use
    - TEDS            # Tree Edit Distance based Similarity
    - Edit_dist       # Normalized Edit Distance
  dataset:                                                                   # Dataset configuration
    dataset_name: omnidocbench_single_module_dataset                         # Dataset name, no need to modify if following specified input format
    ground_truth:                                                            # Configuration for ground truth dataset
      data_path: ./demo_data/recognition/OmniDocBench_demo_table.json        # JSON file containing both ground truth and model prediction results
      data_key: html                                                         # Field name storing Ground Truth, for OmniDocBench, table recognition results are stored in html and latex fields, change to latex when evaluating latex format tables
      category_filter: table                                                 # Category for evaluation, in table recognition, the category_name is table
    prediction:                                                              # Configuration for model prediction results
      data_key: pred                                                         # Field name storing model prediction results, this is user-defined
    category_type: table                                                     # category_type is mainly used for data preprocessing strategy selection
```

For the `dataset` section, the data format in the `ground_truth`'s `data_path` remains consistent with OmniDocBench, with only a custom field added under the corresponding table sample to store the model's prediction result. The field storing prediction information is specified through `data_key` under the `prediction` field in `dataset`, such as `pred`. For more details about OmniDocBench's file structure, please refer to the "Dataset Introduction" section. The input format for model results can be found in [OmniDocBench_demo_table](https://github.com/opendatalab/OmniDocBench/blob/v1_5/demo_data/recognition/OmniDocBench_demo_table.json), which follows this format:

```JSON
[{
    "layout_dets": [    // List of page elements
        {
            "category_type": "table",  // OmniDocBench category name
            "poly": [    // OmniDocBench position info: x,y coordinates for top-left, top-right, bottom-right, bottom-left corners
                136.0, 
                781.0,
                340.0,
                781.0,
                340.0,
                806.0,
                136.0,
                806.0
            ],
            ...   // Other OmniDocBench fields
            "latex": "$xxx$",  // Table LaTeX annotation goes here
            "html": "$xxx$",  // Table HTML annotation goes here
            "pred": "$xxx$",   // !! Model prediction result stored here, user-defined new field at same level as ground truth
            
        ...
    ],
    "page_info": {...},        // OmniDocBench page information
    "extra": {...}             // OmniDocBench annotation relationship information
},
...
]
```

Here is a model inference script for reference:

```PYTHON
import os
import json
from PIL import Image

def poly2bbox(poly):
    L = poly[0]
    U = poly[1]
    R = poly[2]
    D = poly[5]
    L, R = min(L, R), max(L, R)
    U, D = min(U, D), max(U, D)
    bbox = [L, U, R, D]
    return bbox

question = "<image>\nPlease convert this cropped image directly into html format of table."

with open('./demo_data/omnidocbench_demo/OmniDocBench_demo.json', 'r') as f:
    samples = json.load(f)
    
for sample in samples:
    img_name = os.path.basename(sample['page_info']['image_path'])
    img_path = os.path.join('./demo_data/omnidocbench_demo/images', img_name)
    img = Image.open(img_path)

    if not os.path.exists(img_path):
        print('No exist: ', img_name)
        continue

    for i, anno in enumerate(sample['layout_dets']):
        if anno['category_type'] != 'table':   # Filter out the table category for evaluation
            continue

        bbox = poly2bbox(anno['poly'])
        im = img.crop(bbox).convert('RGB')
        response = model.chat(im, question)  # Need to modify the way the image is passed in depending on the model
        anno['pred'] = response              # Directly add a new field to store the model's inference result at the same level as the ground truth

with open('./demo_data/recognition/OmniDocBench_demo_table.json', 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False)
```

</details>



## Layout Detection

OmniDocBench contains bounding box information for all document components on each PDF page, making it suitable as a benchmark for layout detection task evaluation.

<table style="width: 95%; margin: auto; border-collapse: collapse;">
  <thead>
    <tr>
      <th>Model</th>
      <th>Backbone</th>
      <th>Params</th>
      <th>Book</th>
      <th>Slides</th>
      <th>Research<br>Report</th>
      <th>Textbook</th>
      <th>Exam<br>Paper</th>
      <th>Magazine</th>
      <th>Academic<br>Literature</th>
      <th>Notes</th>
      <th>Newspaper</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DiT-L</sup></td>
      <td>ViT-L</td>
      <td>361.6M</td>
      <td><ins>43.44</ins></td>
      <td>13.72</td>
      <td>45.85</td>
      <td>15.45</td>
      <td>3.40</td>
      <td>29.23</td>
      <td><strong>66.13</strong></td>
      <td>0.21</td>
      <td>23.65</td>
      <td>26.90</td>
    </tr>
    <tr>
      <td>LayoutLMv3</sup></td>
      <td>RoBERTa-B</td>
      <td>138.4M</td>
      <td>42.12</td>
      <td>13.63</td>
      <td>43.22</td>
      <td>21.00</td>
      <td>5.48</td>
      <td>31.81</td>
      <td><ins>64.66</ins></td>
      <td>0.80</td>
      <td>30.84</td>
      <td>28.84</td>
    </tr>
    <tr>
      <td>DocLayout-YOLO</sup></td>
      <td>v10m</td>
      <td>19.6M</td>
      <td><strong>43.71</strong></td>
      <td><strong>48.71</strong></td>
      <td><strong>72.83</strong></td>
      <td><strong>42.67</strong></td>
      <td><strong>35.40</strong></td>
      <td><ins>51.44</ins></td>
      <td><ins>64.64</ins></td>
      <td><ins>9.54</ins></td>
      <td><strong>57.54</strong></td>
      <td><strong>47.38</strong></td>
    </tr>
    <tr>
      <td>SwinDocSegmenter</sup></td>
      <td>Swin-L</td>
      <td>223M</td>
      <td>42.91</td>
      <td><ins>28.20</ins></td>
      <td><ins>47.29</ins></td>
      <td><ins>32.44</ins></td>
      <td><ins>20.81</ins></td>
      <td><strong>52.35</strong></td>
      <td>48.54</td>
      <td><strong>12.38</strong></td>
      <td><ins>38.06</ins></td>
      <td><ins>35.89</ins></td>
    </tr>
    <tr>
      <td>GraphKD</sup></td>
      <td>R101</td>
      <td>44.5M</td>
      <td>39.03</td>
      <td>16.18</td>
      <td>39.92</td>
      <td>22.82</td>
      <td>14.31</td>
      <td>37.61</td>
      <td>44.43</td>
      <td>5.71</td>
      <td>23.86</td>
      <td>27.10</td>
    </tr>
    <tr>
      <td>DOCX-Chain</sup></td>
      <td>-</td>
      <td>-</td>
      <td>30.86</td>
      <td>11.71</td>
      <td>39.62</td>
      <td>19.23</td>
      <td>10.67</td>
      <td>23.00</td>
      <td>41.60</td>
      <td>1.80</td>
      <td>16.96</td>
      <td>21.27</td>
    </tr>
  </tbody>
</table>

<p>Component-level layout detection evaluation on OmniDocBench (v1.0) layout subset: mAP results by PDF page type.</p>



Layout detection config file reference [layout_detection](https://github.com/opendatalab/OmniDocBench/blob/v1_5/configs/layout_detection.yaml), data format reference [detection_prediction](https://github.com/opendatalab/OmniDocBench/blob/v1_5/demo_data/detection/detection_prediction.json).

<details>
  <summary>Field explanation: layout_detection.yaml</summary>

Here is the configuration file for `layout_detection.yaml`:

```YAML
detection_eval:   # Specify task name, common for all detection-related tasks
  metrics:
    - COCODet     # Detection task related metrics, mainly mAP, mAR etc.
  dataset: 
    dataset_name: detection_dataset_simple_format       # Dataset name, no need to modify if following specified input format
    ground_truth:
      data_path: ./demo_data/omnidocbench_demo/OmniDocBench_demo.json               # Path to OmniDocBench JSON file
    prediction:
      data_path: ./demo_data/detection/detection_prediction.json                    # Path to model prediction result JSON file
    filter:                                             # Page level filtering
      data_source: exam_paper                           # Page attributes and corresponding tags to be evaluated
  categories:
    eval_cat:                # Categories participating in final evaluation
      block_level:           # Block level categories, see OmniDocBench evaluation set introduction for details
        - title              # Title
        - text               # Text  
        - abandon            # Includes headers, footers, page numbers, and page annotations
        - figure             # Image
        - figure_caption     # Image caption
        - table              # Table
        - table_caption      # Table caption
        - table_footnote     # Table footnote
        - isolate_formula    # Display formula (this is a layout display formula, lower priority than 14)
        - formula_caption    # Display formula label
    gt_cat_mapping:          # Mapping table from ground truth to final evaluation categories, key is ground truth category, value is final evaluation category name
      figure_footnote: figure_footnote
      figure_caption: figure_caption 
      page_number: abandon 
      header: abandon 
      page_footnote: abandon
      table_footnote: table_footnote 
      code_txt: figure 
      equation_caption: formula_caption 
      equation_isolated: isolate_formula
      table: table 
      refernece: text 
      table_caption: table_caption 
      figure: figure 
      title: title 
      text_block: text 
      footer: abandon
    pred_cat_mapping:       # Mapping table from prediction to final evaluation categories, key is prediction category, value is final evaluation category name
      title : title
      plain text: text
      abandon: abandon
      figure: figure
      figure_caption: figure_caption
      table: table
      table_caption: table_caption
      table_footnote: table_footnote
      isolate_formula: isolate_formula
      formula_caption: formula_caption
```

The `filter` field can be used to filter the dataset. For example, setting the `filter` field under `dataset` to `data_source: exam_paper` will filter for pages with data type exam_paper. For more page attributes, please refer to the "Evaluation Set Introduction" section. If you want to evaluate the full dataset, comment out the `filter` related fields.

The `data_path` under the `prediction` section in `dataset` takes the model's prediction as input, with the following data format:

```JSON
{
    "results": [
        {
            "image_name": "docstructbench_llm-raw-scihub-o.O-adsc.201190003.pdf_6",                     // image name
            "bbox": [53.892921447753906, 909.8675537109375, 808.5555419921875, 1006.2714233398438],     // bounding box coordinates, representing x,y coordinates of top-left and bottom-right corners
            "category_id": 1,                                                                           // category ID number
            "score": 0.9446213841438293                                                                 // confidence score
        }, 
        ...                                                                                             // all bounding boxes are flattened in a single list
    ],
    "categories": {"0": "title", "1": "plain text", "2": "abandon", ...}                                // mapping between category IDs and category names
```

</details>


## Evaluated Models and Sources

### Text Recognition Models

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Official Website</th>
      <th>Evaluation Version/Model Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PaddleOCR</td>
      <td><a href="https://www.paddlepaddle.org.cn/hub/scene/ocr">PaddlePaddle OCR</a></td>
      <td>2.9.1</td>
    </tr>
    <tr>
      <td>Tesseract</td>
      <td><a href="https://tesseract-ocr.github.io/tessdoc/">Tesseract OCR</a></td>
      <td>5.5</td>
    </tr>
    <tr>
      <td>OpenOCR</td>
      <td><a href="https://github.com/Topdu/OpenOCR">OpenOCR GitHub</a></td>
      <td>0.0.6</td>
    </tr>
    <tr>
      <td>EasyOCR</td>
      <td><a href="https://www.easyproject.cn/easyocr/">EasyOCR</a></td>
      <td>1.7.2</td>
    </tr>
    <tr>
      <td>Surya</td>
      <td><a href="https://github.com/VikParuchuri/surya">Surya GitHub</a></td>
      <td>0.5.0</td>
    </tr>
  </tbody>
</table>


### Layout Models

<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Official Website</th>
      <th>Evaluation Version/Model Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DiT-L</td>
      <td><a href="https://github.com/facebookresearch/DiT">DiT-L</a></td>
      <td><a href="https://huggingface.co/docs/transformers/model_doc/dit">Hugging Face DiT</a></td>
    </tr>
    <tr>
      <td>LayoutMv3</td>
      <td><a href="https://github.com/microsoft/unilm/tree/master/layoutlmv3">LayoutMv3</a></td>
      <td><a href="https://huggingface.co/docs/transformers/model_doc/layoutlmv3">Hugging Face LayoutMv3</a></td>
    </tr>
    <tr>
      <td>DOCX-Chain</td>
      <td><a href="https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Applications/DocXChain">DOCX-Chain</a></td>
      <td><a href="https://github.com/AlibabaResearch/AdvancedLiterateMachinery/releases/download/v1.2.0-docX-release/DocXLayout_231012.pth">DocXLayout_231012.pth</a></td>
    </tr>
    <tr>
      <td>DocLayout-YOLO</td>
      <td><a href="https://github.com/opendatalab/DocLayout-YOLO">DocLayout-YOLO</a></td>
      <td><a href="https://huggingface.co/spaces/opendatalab/DocLayout-YOLO">Hugging Face DocLayout-YOLO</a></td>
    </tr>
    <tr>
      <td>SwinDocSegmenter</td>
      <td><a href="https://github.com/ayanban011/SwinDocSegmenter">SwinDocSegmenter</a></td>
      <td><a href="https://drive.google.com/file/d/1DCxG2MCza_z-yB3bLcaVvVR4Jik00Ecq/view?usp=share_link">model weights</a></td>
    </tr>
    <tr>
      <td>GraphKD</td>
      <td><a href="https://github.com/ayanban011/GraphKD">GraphKD</a></td>
      <td><a href="https://drive.google.com/file/d/1oOzy7D6J0yb0Z_ALwpPZMbIZf_AmekvE/view?usp=sharing">model weights</a></td>
    </tr>
  </tbody>
</table>


### Table Structure Models
<table>
  <thead>
    <tr>
      <th>Model Name</th>
      <th>Official Website</th>
      <th>Evaluation Version/Model Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PaddleOCR</td>
      <td><a href="https://github.com/PaddlePaddle/PaddleOCR">PaddleOCR</a></td>
      <td><a href="https://paddlepaddle.github.io/PaddleOCR/latest/model/index.html">PaddleOCR</a></td>
    </tr>
    <tr>
      <td>RapidTable</td>
      <td><a href="https://github.com/RapidAI/RapidTable">RapidTable</a></td>
      <td><a href="https://www.modelscope.cn/models/RapidAI/RapidTable/files">ModelScope RapidTable</a></td>
    </tr>
    <tr>
      <td>StructEqTable</td>
      <td><a href="https://github.com/Alpha-Innovator/StructEqTable-Deploy/blob/main/README.md">StructEqTable</a></td>
      <td><a href="https://huggingface.co/U4R/StructTable-base">Hugging Face StructEqTable</a></td>
    </tr>
    <tr>
      <td>GOT-OCR</td>
      <td><a href="https://github.com/Ucas-HaoranWei/GOT-OCR2.0">GOT-OCR</a></td>
      <td><a href="https://huggingface.co/stepfun-ai/GOT-OCR2_0">Hugging Face GOT-OCR</a></td>
    </tr>
    <tr>
      <td>Qwen2-VL-7B</td>
      <td><a href="https://github.com/QwenLM/Qwen2-VL">Qwen2-VL-7B</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct">Hugging Face Qwen2-VL-7B-Instruct</a></td>
    </tr>
    <tr>
      <td>InternVL2-8B</td>
      <td><a href="https://github.com/OpenGVLab/InternVL">InternVL2-8B</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">Hugging Face InternVL2-8B</a></td>
    </tr>
  </tbody>
</table>

