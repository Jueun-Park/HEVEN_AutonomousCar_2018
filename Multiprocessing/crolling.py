# parser.py
import requests
from bs4 import BeautifulSoup as bs
import time

from multiprocessing import Pool # Pool import하기


def get_links(): # 블로그의 게시글 링크들을 가져옵니다.cmd
    req = requests.get('https://beomi.github.io/beomi.github.io_old/')
    html = req.text
    soup = bs(html, 'html.parser')
    my_titles = soup.select(
        'h3 > a'
        )
    data = []

    for title in my_titles:
        data.append(title.get('href'))
    return data

def get_content(link):
    abs_link = 'https://beomi.github.io'+link
    req = requests.get(abs_link)
    html = req.text
    soup = bs(html, 'html.parser')
    # 가져온 데이터로 뭔가 할 수 있겠죠?
    # 하지만 일단 여기서는 시간만 확인해봅시다.
    print(soup.select('h1')[0].text) # 첫 h1 태그를 봅시다.

if __name__=='__main__':
    start_time = time.time()
    pool = Pool(processes=10) # 4개의 프로세스를 사용합니다.
    pool.map(get_content, get_links()) # get_contetn 함수를 넣어줍시다.
    map()
    print("--- %s seconds ---" % (time.time() - start_time))