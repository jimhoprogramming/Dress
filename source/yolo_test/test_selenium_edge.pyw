from selenium import webdriver

driver = webdriver.Edge()
driver.get("https://Trade2.guosen.com.cn")

cookies=driver.get_cookie('LTOKEN')
#cookies=driver.get_cookies()
print(cookies)

driver.quit()
print(cookies['value'])

