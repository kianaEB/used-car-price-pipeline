# ============================================================================
# PRESERVED PROTOTYPE - cars_original.py  (DO NOT RUN OR EXTEND)
# ----------------------------------------------------------------------------
# This is the original 207-line script (github.com/kianaEB/Cars_Price_prediction_by_ML),
# kept ONLY for the before/after portfolio story. The rebuilt pipeline lives in src/.
#
# The plaintext MySQL password was REDACTED below (the original hardcoded it).
# >>> Rotate that password anywhere it was ever reused. <<<
#
# Known flaws (all fixed in the rebuild - see SPEC.md "Motivation"):
#   1. DecisionTreeClassifier used on a CONTINUOUS price target (should be a regressor).
#   2. No train/test split and NO evaluation metric is ever computed.
#   3. Mileage is overwritten to 0 for every training row (dead feature).
#   4. Convoluted LabelEncoder-via-temp-CSV encoding.
#   5. Brittle hardcoded CSS selectors; no politeness/rate limiting.
#   6. Secrets committed in source; everything in one top-level script; no tests.
# ============================================================================

import requests
import time
from bs4 import BeautifulSoup
import mysql.connector
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#####################################  Write & Read Temp CSV  #######################################
def writeTempCSV(brands, models):
    data = list()
    header = ["Brand", "Model"]

    for i in range(len(brands)):
        data.append([brands[i], models[i]])

    with open('temp.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    
def readTempCSV(): 
    dataframe = pd.read_csv('temp.csv')   
    le = LabelEncoder()
    brandLabel = le.fit_transform(dataframe['Brand'])
    modelLabel = le.fit_transform(dataframe['Model'])
    
    return brandLabel, modelLabel

#####################################  Write & Read Car CSV  #######################################
def writeCarsCSV(brands, models, dates, miles, prices):
    data = list()

    header = ["Brand", "Model", "Year", "Mile", "Price"]

    for i in range(len(brands)):
        data.append([brands[i], models[i], dates[i], miles[i], prices[i]])

    brandChoice = data[len(brands) - 1][0]
    modelChoice = data[len(brands) - 1][1]
    yearChoice  = data[len(brands) - 1][2]
    mileChoice  = data[len(brands) - 1][3]

    del(data[len(brands) - 1])

    with open('cars.csv', 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    
    return brandChoice, modelChoice, yearChoice, mileChoice     

###################################### Price Prediction  #########################################

def predictPrice(brandChoice, modelChoice, yearChoice, mileChoice):
    data = pd.read_csv("cars.csv")

    x = data.drop(columns = ["Price"])
    y = data["Price"]

    model = DecisionTreeClassifier()
    model.fit(x.values, y)

    prediction = model.predict([[brandChoice, modelChoice, yearChoice, mileChoice]])

    print("The price could be $%i" % prediction)



cnx = mysql.connector.connect(user = 'root', password = 'REDACTED-ROTATE-THIS-PASSWORD', host = '127.0.0.1', database = 'jadi')
cursor = cnx.cursor()

cursor = cnx.cursor(buffered=True)
cursor.execute("SHOW TABLES;")
if ('cars',) in cursor:
    cursor.execute('DROP TABLE cars;')
cursor.execute('CREATE TABLE cars ( brand VARCHAR(100), model VARCHAR(100), date VARCHAR(100), price VARCHAR(100), miles VARCHAR(100));') 

k = 0
dates = list()
brands = list()
models = list()
miles = list()
prices = list()

for i in range(1, 50):
    page = ''
    while page == '':
        try:
            page = requests.get("https://www.cars.com/shopping/results/?page=" + str(i) + "&page_size=20&dealer_id=&keyword=&list_price_max=&list_price_min=&makes[]=&maximum_distance=all&mileage_max=&sort=best_match_desc&stock_type=new&year_max=&year_min=&zip=")
            break
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            time.sleep(5)
            print("Now let me continue...")
            continue
    
    soup = BeautifulSoup(page.text, 'html.parser')
    string = soup.find_all(class_="title")
    prices_count = soup.find_all(class_="primary-price")
    j = 0
    for one in string:
        price = ""
        if prices_count[j].text != 'Not Priced':
            for n in range(len(prices_count[j].text)):
                if prices_count[j].text[n] >= '0' and prices_count[j].text[n] <= '9':
                    price += prices_count[j].text[n]
            prices.append(price)
        else:
            j += 1
            continue       
        finding = one.text.split()
        dates.append(finding[0])
        brands.append(finding[1])
        models.append(' '.join(finding[2:]))
        miles.append('0')

        query = ("INSERT INTO cars "
        "(brand, model, date, price, miles)"
        "VALUES (%s, %s, %s, %s, %s);")
        cursor.execute(query, (brands[k], models[k], dates[k], prices[k], miles[k]))
        
        miles[k] = 0
        prices[k] = int(price)
        dates[k] = int(finding[0])        
        
        j += 1
        k += 1
    
for i in range(1, 200):  
    page = ''
    while page == '':
        try:
            page = requests.get("https://www.cars.com/shopping/results/?page=" + str(i) + "&page_size=20&list_price_max=&makes[]=&maximum_distance=all&models[]=&stock_type=used&zip=")
            break
        except:
            print("Connection refused by the server..")
            print("Let me sleep for 5 seconds")
            time.sleep(5)
            print("Now let me continue...")
            continue
    
    soup = BeautifulSoup(page.text, 'html.parser')  
    string = soup.find_all(class_="title")
    prices_count = soup.find_all(class_="primary-price")
    miles_count = soup.find_all(class_="mileage")
    j = 0
    
    for one in string:
        mile = ""
        if miles_count[j].text[0] >= '0' and miles_count[j].text[0] <= '9' and prices_count[j].text != 'Not Priced':
            for n in range(len(miles_count[j].text)):
                if miles_count[j].text[n] >= '0' and miles_count[j].text[n] <= '9':
                    mile += miles_count[j].text[n]
            miles.append(mile)        
        else:
            j += 1 
            continue          
        
        finding = one.text.split()
        dates.append(finding[0])
        brands.append(finding[1])
        models.append(' '.join(finding[2:]))
        price = ""
        for n in range(len(prices_count[j].text)):
            if prices_count[j].text[n] >= '0' and prices_count[j].text[n] <= '9':
                price += prices_count[j].text[n]
        prices.append(price)        

        query = ("INSERT INTO cars "
        "(brand, model, date, price, miles)"
        "VALUES (%s, %s, %s, %s, %s);")
        cursor.execute(query, (brands[k], models[k], dates[k], prices[k], miles[k]))
        
        miles[k] = 0
        prices[k] = int(price)
        dates[k] = int(finding[0])              
        
        j += 1
        k += 1
        
cnx.commit()
cursor.close()
cnx.close()    
    
###################################### Prediction Time  ######################################### 

brand = input("Please input a brand: ")
mod = input("Please input a model: ")
year = int(input("Please input a year: "))
mile = int(input("Please input a mileage: "))       

brands.append(brand)
models.append(mod)
dates.append(year)
miles.append(mile)
prices.append(0)

writeTempCSV(brands, models)
brands, models = readTempCSV()
brandChoice, modelChoice, yearChoice, mileChoice = writeCarsCSV(brands, models, dates, miles, prices)

predictPrice(brandChoice, modelChoice, yearChoice, mileChoice)
      
    