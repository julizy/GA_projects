# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2: Singapore Housing Price Prediction

## Table of Contents
- [Background](#Background)  
- [Problem Statement](#Problem-Statement) 
- [Data Dictionary](#Data-Dictionary)
- [Executive Summary](#Executive-Summary)
- [Conclusion and Recommendation](#Conclusion-and-Recommendation)  

## Background
HDB (Housing and Development Board) flats are public housing units in Singapore, constructed and subsidized by the government to provide affordable housing options to citizens. They are a popular choice for first-time homebuyers in the country. There are two options for purchasing an HDB flat: applying for a new Build-To-Order (BTO) unit or buying a resale HDB flat from the open market.

Resale HDB flats are a convenient option for those who don't want to wait for BTO units to be built as they typically take several years to complete. However, resale flats tend to be more expensive compared to BTO units which are sold at the prices below market value by HDB. It is important to estimate the resale price of HDB flat in the open market to ensure a fair deal for both the buyer and seller. Both parties can benefit from having a sense of the fair market value for the HDB flat.

<img src="images/image_1.PNG" width="800"/>

[Image Source](https://blog.seedly.sg/which-should-you-get-bto-vs-resale/)

[Return to top](#Table-of-Contents)  

## Problem Statement
As data scientists working for a real estate agent, our goal is to empower prospective householders and sellers to make informed decisions about property purchases and sales. Our main objectives are:

**- Use Singapore public housing data from 2012 to 2021 to create a regression model that predicts the resale price of HDB flats in Singapore.**

**- Identify the top 5 features that have the greatest positive correlation to the resale price so that buyers and sellers can take this information into consideration.**

Data files & materials: [DSI-SG-Project-2 Regression Challenge](https://www.kaggle.com/competitions/dsi-sg-project-2-regression-challenge-hdb-price/overview)

[Return to top](#Table-of-Contents)  

## Data Dictionary
The final data has 95 columns (94 features + 1 target variable) which contains 167371 HDB flat resale records (150634 for train set and 16737 for test set) from 2012 to 2021.
|Feature|Type|Description| 
|---|---|---|
|resale_price|float|property's resale price in Singapore dollars. This is the target variable that we are trying to predict| 
|town|uint|HDB township where the flat is located, e.g. BUKIT MERAH (one-hot encoded, 26 features)| 
|flat_model|uint|HDB model of the resale flat, e.g. Multi Generation (one-hot encoded, 20 features)| 
|flat_type|string|type of the resale flat unit, e.g. 3 ROOM (7 types in total)| 
|tranc_year|int|year of resale transaction|
|tranc_month|int|month of resale transaction| 
|mid_storey|int|median value of storey_range| 
|floor_area_sqft|float|floor area of the resale flat unit in square feet| 
|hdb_age|int|number of years from lease_commence_date to present year| 
|max_floor_lvl|int|highest floor of the resale flat| 
|year_completed|int|year which construction was completed for resale flat| 
|commercial|int|boolean value (0/1) if resale flat has commercial units in the same block| 
|market_hawker|int|boolean value (0/1) if resale flat has a market or hawker centre in the same block|
|multistorey_carpark|int|boolean value (0/1) if resale flat has a multistorey carpark in the same block|
|precinct_pavilion|int|boolean value (0/1) if resale flat has a pavilion in the same block| 
|total_dwelling_units|int|total number of residential dwelling units in the resale flat| 
|1room_sold|int|number of 1-room residential units in the resale flat| 
|2room_sold|int|number of 2-room residential units in the resale flat| 
|3room_sold|int|number of 3-room residential units in the resale flat| 
|4room_sold|int|number of 4-room residential units in the resale flat| 
|5room_sold|int|number of 5-room residential units in the resale flat| 
|exec_sold|int|number of executive type residential units in the resale flat block| 
|multigen_sold|int|number of multi-generational type residential units in the resale flat block|
|studio_apartment_sold|int|number of studio apartment type residential units in the resale flat block| 
|1room_rental|int|number of 1-room rental residential units in the resale flat block| 
|2room_rental|int|number of 2-room rental residential units in the resale flat block| 
|3room_rental|int|number of 3-room rental residential units in the resale flat block| 
|other_room_rental|int|number of "other" type rental residential units in the resale flat block|
|mall_nearest_distance|float|distance (in metres) to the nearest mall| 
|mall_within_500m|int|number of malls within 500 metres| 
|mall_within_1km|int|number of malls within 1 kilometre| 
|mall_within_2km|int|number of malls within 2 kilometres| 
|hawker_nearest_distance|float|distance (in metres) to the nearest hawker centre| 
|hawker_within_500m|int|number of hawker centres within 500 metres| 
|hawker_within_1km|int|number of hawker centres within 1 kilometre| 
|hawker_within_2km|int|number of hawker centres within 2 kilometres|
|hawker_food_stalls|int|number of hawker food stalls in the nearest hawker centre|
|hawker_market_stalls|int|number of hawker and market stalls in the nearest hawker centre|
|mrt_nearest_distance|float|distance (in metres) to the nearest MRT station|
|bus_interchange|int|boolean value (0/1) if the nearest MRT station is also a bus interchange|
|mrt_interchange|int|boolean value (0/1) if the nearest MRT station is a train interchange station|
|bus_stop_nearest_distance|float|distance (in metres) to the nearest bus stop|
|pri_sch_nearest_distance|float|distance (in metres) to the nearest primary school|
|vacancy|int|number of vacancies in the nearest primary school|
|pri_sch_affiliation|int|boolean value (0/1) if the nearest primary school has a secondary school affiliation|
|sec_sch_nearest_dist|float|distance (in metres) to the nearest secondary school|
|cutoff_point|float|PSLE cutoff point of the nearest secondary school|
|affiliation|int|boolean value (0/1) if the nearest secondary school has an primary school affiliation|
|mall_500m_to_1km|int|number of malls between 500 metres to 1km|              
|mall_1km_to_2km|int|number of malls between 1km to 2km|               
|hawker_500m_to_1km|int|number of hawkers between 500 metres to 1km|              
|hawker_1km_to_2km|int|number of hawkers between 1km to 2km|              
|age_of_built|int|number of years from tranc_year to year_completed|

[Return to top](#Table-of-Contents)

## Executive Summary

### EDA

### Modeling

[Return to top](#Table-of-Contents)

## Conclusion and Recommendation  

### Conclusion
We built 7 models for each flat type to predict the resale price, in overall with a **R2 score** of **0.928** and **RMSE** of **39,415.31**. Besides, we were able to identify the **top 5 features** that were **positively correlated** to the **resale price** for **each flat type**, as shown in the results below: 

|   |1 Room|2 Room|3 Room|4 Room|5 Room|Exec|Multi|
|---|---|---|---|---|---|---|---|
|**1st**|tranc_month|4room_sold|max_floor_lvl|max_floor_lvl|hawker_within_2km|hawker_within_2km|hawker_within_2km|
|**2nd**|mid_storey|floor_area_sqft|floor_area_sqft|hawker_within_2km|max_floor_lvl|hawker_within_1km|hawker_market_stalls|
|**3rd**|tranc_year|max_floor_lvl|mid_storey|hawker_within_1km|hawker_within_1km|floor_area_sqft|hawker_food_stalls|
|**4th**|-|mid_storey|hawker_within_2km|mid_storey|hawker_within_500m|hdb_age|hawker_within_1km|
|**5th**|-|mall_within_500m|mall_within_2km|hawker_within_500m|mid_storey|affiliation|floor_area_sqft|

Generally speaking, **max floor level**, **number of hawkers/malls within a certain distance**, **floor area** and **mid storey** are the top positively correlated features, however the order of correlation varies depending on different flat types. 

### Recommendation
When considering purchasing an HDB resale flat, it's important to keep in mind that in general the **floor area** is typically the **most** significant factor affecting the resale price. Historical data shows same as common sense: larger flats tend to be more expensive. However, to get a more accurate estimate of what you can expect to pay, it's advisable to dive deeper into HDB transactions and pricing trends for **different flat types**.

For example, if you're looking to buy a **3 Room** or **4 Room** flat, you should pay close attention to the **maximum floor level** of the building as flats with higher maximum floor levels are often more expensive. On the other hand, if you're considering a **5 Room**, **Executive**, or **Multi-generation** flat, the **number of hawkers within 2km** is a key factor to consider. Historical data suggests that flats located in areas with more hawkers within 2km tend to command higher prices.

As a seller of an HDB resale flat, taking into account the information mentioned above can help you to maximize the value of your sale.

[Return to top](#Table-of-Contents)  

