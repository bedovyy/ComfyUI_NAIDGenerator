"""
Wildcards for NAI
Written by anonymous user from arca.live
Modified some code
@URL <TBD>
"""
import os, glob
import random
import re
import os
import fnmatch
import chardet
from typing import Dict, List

class wildcards:

    # 가져올 파일 목록
    card_path = os.path.join(os.path.dirname(__file__), "wildcards", "*.txt")
    #card_path=f"{os.getcwd()}\\wildcards\\**\\*.txt"
    print(f"wildcards card_path : ", card_path)

    # 정규식
    #resub  = re.compile(r"(\{)([^\{\}]*)(\})")
    #resub  = re.compile(r"(\{)(((\d+)|(\d+)?-(\d+)?)?\$\$((.*)?\$\$)?)?([^\{\}]*)(\})")
    resub  = re.compile(r"(\{)(((\d+)|(\d+)?-(\d+)?)?\$\$(([^\{\}]*?)\$\$)?)?([^\{\}]*)(\})")
    recard = re.compile(r"(__)(.*?)(__)")

    # 카드 목록
    is_card_Load = False
    cards: Dict[str, List[str]]= {}
    seperator=", "
    loop_max=50

    # | 로 입력된것중 하나 가져오기
    def sub(match):
        #print(f"sub : {(match.groups())}")
        try:        
            #m=match.group(2)
            seperator=wildcards.seperator
            s=match.group(3)
            m=match.group(9).split("|")
            p=match.group(8)
            if p:
                seperator=p
                
            if s is None:
                return random.choice(m)
            c=len(m)
            n=int(match.group(4)) if  match.group(4) else None
            if n:

                r=seperator.join(random.sample(m,min(n,c)))
                #print(f"n : {n} ; {r}")
                return r

            n1=match.group(5)
            n2=match.group(6)
            
            if n1 or n2:
                a=min(int(n1 if n1 else c), int(n2 if n2 else c),c)
                b=min(max(int(n1 if n1 else 0), int(n2 if n2 else 0)),c)
                #print(f"ab : {a} ; {b}")
                r=seperator.join(
                    random.sample(
                        m,
                        random.randint(
                            a,b
                        )
                    )
                )
                #n1=int(match.group(5)) if not match.group(5) is None 
                #n2=int(match.group(6)) if not match.group(6) is None 
            else:
                r=seperator.join(
                    random.sample(
                        m,
                        random.randint(
                            0,c
                        )
                    )
                )
            #print(f"12 : {r}")
            return r


        except Exception as e:         
            console.print_exception()
            return ""
            
            

    # | 로 입력된것중 하나 가져오기 반복
    def sub_loop(text):
        """
        selects from {a|b|c} style
        """
        if "|" not in text:
            return text
        target_text=text
        for i in range(1, wildcards.loop_max):
            tmp=wildcards.resub.sub(wildcards.sub, target_text)
            #print(f"tmp : {tmp}")
            if target_text==tmp :
                return tmp
            target_text=tmp
        return target_text

    # 카드 중에서 가져오기
    def card(match: re.Match):
        """
        Find the __card__ in the text and replace it with a random card value
        """
        #print(f"card in  : {match.group(2)}")
        card_lst=fnmatch.filter(wildcards.cards, match.group(2))
        if len(card_lst)>0:
            #print(f"card lst : {lst}")
            cd=random.choice(card_lst)
            #print(f"card get : {cd}")
            r=random.choice(wildcards.cards[cd])        
        else :    
            r= match.group(2)
        #print(f"card out : {r}")
        return r
        

    # 카드 중에서 가져오기 반복. | 의 것도 처리
    def card_loop(text):
        """
        Main entry point for the application script
        """
        target_text=text
        for i in range(1, wildcards.loop_max):
            tmp=wildcards.recard.sub(wildcards.card, target_text)
            print(f"card deck selected : {tmp}")
            if target_text==tmp :
                # failed to find card
                tmp=wildcards.sub_loop(tmp)
                
            if target_text==tmp :
                #print(f"card le : {target_text}")
                return tmp
            target_text=tmp
        #print(f"card le : {target_text}")
        return target_text
        
    # 카드 파일 읽기
    def card_load():
        #cards=wildcards.cards
        card_path=wildcards.card_path
        cards = {}
        #print(f"path : {path}")
        files=glob.glob(card_path, recursive=True)
        #print(f"files : {files}")
        
        for file in files:
            basenameAll = os.path.basename(file)
            basename = os.path.relpath(file, os.path.dirname(__file__)).replace("\\", "/").replace("../../wildcards/", "")
            #print(f"basenameAll : {basenameAll}")
            #print(f"basename : {basename}")
            file_nameAll = os.path.splitext(basenameAll)[0]
            file_name = "/"+os.path.splitext(basename)[0]
            #print(f"file_nameAll : {file_nameAll}")
            #print(f"file_name : {file_name}")
            if not file_nameAll in cards:
                cards[file_nameAll]=[]
            if not file_name in cards:
                cards[file_name]=[]
            #print(f"file_name : {file_name}")
            with open(file, "rb") as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)["encoding"]
            with open(file, "r", encoding=encoding) as f:
                lines = f.readlines()
                for line in lines:
                    line=line.strip()
                    # 주석 빈줄 제외
                    if line.startswith("#") or len(line)==0:
                        continue
                    cards[file_nameAll]+=[line]
                    cards[file_name]+=[line]
                    #print(f"line : {line}")
            print(f"card file : {file_nameAll} ; {len(cards[file_nameAll])}")
        wildcards.cards=cards
        print(f"cards file count : ", len(wildcards.cards))
        #print(f"cards : {cards.keys()}")
        wildcards.is_card_Load=True

    # 실행기
    def run(text,load=False):
        if text is None or not isinstance(text, str):
            print("text is not str : ",text)
            return None
        if not wildcards.is_card_Load or load:
            wildcards.card_load()

        print(f"text : {text}")
        result=wildcards.card_loop(text)
        print(f"result : {result}")
        return result

def SD2NAIstyle(wild):
    wild=wild.replace("\(","▤")
    wild=wild.replace("\)","▥")
    wild=wild.replace("(","{")
    wild=wild.replace(")","}")
    wild=wild.replace("▤", "(")
    wild=wild.replace("▥", ")")
    return wild


    

class NAITextWildcards:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "SD2NAI": (["none", "SD2NAI text style"], {"default":"SD2NAI text style"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            
        }
        }
    RETURN_TYPES = ("STRING","ASCII")
    FUNCTION = "encode"

    CATEGORY = "NAI"

    def encode(self, seed, text, SD2NAI):
        random.seed(seed)
        # print(f"original text : ",text)
        r=wildcards.run(text)
        # print(f"wildcard result : ",r)
        if SD2NAI == "SD2NAI text style":
            r=SD2NAIstyle(r)
            # print(f"SD2NAI result : ",r)
        return (r, r)
