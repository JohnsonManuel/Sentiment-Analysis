from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def get_reviews(driver):
    review_list = []
    reviews = driver.find_elements(By.CLASS_NAME,"review")
    try:
        for items in reviews:
            review_list.append(items.find_element(By.CLASS_NAME,"review-text").text.strip())
    except Exception as e:
        print(e)
    return review_list


def start_scraping(url,page_count=10):
    final_review_set=[]
    url = url.replace("dp","product-reviews")
    driver = webdriver.Firefox() #webdriver.Chrome()
    loc = url.find("ref")
    initial_page_ref = "ref=cm_cr_dp_d_show_all_btm"
    url_final = url[:loc]+f"/{initial_page_ref}?ie=UTF8&reviewerType=all_reviews&pageNumber="
    new_page = url_final + f'1'
    for x in range(2,page_count):
        driver.get(new_page)
        time.sleep(5)
        reviews = get_reviews(driver)
        #check if there is a next page
        try:
            next_page = driver.find_element(By.CLASS_NAME,'a-disabled a-last')
            break
        except:
            pass
        paging_ref = f'cm_cr_arp_d_paging_btm_next_{x}'
        new_page = url[:loc]+f"/{paging_ref}?ie=UTF8&reviewerType=all_reviews&pageNumber={x}"
        final_review_set.extend(reviews)

    driver.close()
    return final_review_set

#https://www.amazon.in/Apple-iPhone-15-Pro-128/product-reviews/B0CHX7J4TL/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews

if __name__=="__main__":
    url = "https://www.amazon.in/Apple-iPhone-15-Pro-128/dp/B0CHX7J4TL/?_encoding=UTF8&pd_rd_w=nnZjH&content-id=amzn1.sym.44901b9b-bd56-4240-8b6b-3ad72079fb43%3Aamzn1.symc.adba8a53-36db-43df-a081-77d28e1b71e6&pf_rd_p=44901b9b-bd56-4240-8b6b-3ad72079fb43&pf_rd_r=0MS16KJX9DVCT763A433&pd_rd_wg=IECeA&pd_rd_r=5ac44d00-8430-4614-afd9-9eacc1fb642c&ref_=pd_gw_ci_mcx_mr_hp_atf_m&th=1"
    final_review_set = start_scraping(url,5)
    print(final_review_set)

