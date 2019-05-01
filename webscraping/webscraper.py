from bs4 import BeautifulSoup
import requests
import json


def parse_content(content, output_file):
    if not content:
        return
    soup = BeautifulSoup(content, features="html.parser")
    output = []
    for question in soup.find_all('a', {"class": "mod_content_accordion_title_link"}):
        output.append({"question": question.text})

    answers_div = soup.find_all(
        "div", {"class": "mod_content_accordion_tab_panel"})

    for i, a_div in enumerate(answers_div):
        standard_content = a_div.div
        mod_text = standard_content.div
        answer_p = mod_text.p

        obj = output[i]
        obj["answer"] = answer_p.text
        output[i] = obj

    with open(output_file, "w") as fp:
        json.dump(output, fp)


def parse(url, output_file):
    response = requests.get(url)
    if response.status_code == 200:
        print("successfully requested. Length is ", len(response.text))
        parse_content(response.text, output_file)
    else:
        print("failed to query ", url, response.status_code)
