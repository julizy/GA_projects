# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone Project: Image Clustering for Auto Annotation

## Table of Contents
- [Background](#Background)  
- [Problem Statement](#Problem-Statement) 
- [Data Source](#Data-Source)
- [Challenges and Proposed Solution](#Challenges-and-Proposed-Solution)  
- [Audience and Stakeholders](#Audience-and-Stakeholders)

## Background
Automated-optical-inspection (AOI) tools are deployed prior to shipment of ICs as a quality gate. These tools rely on rule-based algorithms that have low classification accuracy. A solution based on Deep Learning models has been developed to improve the optical inspection process. However, so far all the labelling of the images done manually, which is time consuming and costly. 

**Will explain more about the business concept, end goal and impact in the final version**

[Return to top](#Table-of-Contents)  

## Problem Statement
Using image clustering for auto annotation to reduce the time and cost for manual labelling of defect IC packages (e.g. VQ48). 

[Return to top](#Table-of-Contents)  

## Data Source
Company in-house data (e.g. KLA images for VQ48)

[Return to top](#Table-of-Contents)

## Challenges and Proposed Solution
### Challenges
- How to standardize the images (different machine/lighting, image size, etc.) and if use original package images, need to find how to locate the defect positions 
- Processed data as an alternative: script to crop the image at specific zoom-in defect area 
- Clustering with different failure modes (crack line/ single point defect)
- Success metrics (nmi, acc, ari usual metrics for clustering, for our case it needs to check if the results have consistent patterns of failure modes)
  
### Proposed Solution
- Transfer learning CNN + clustering (e.g. K-means) as baseline
- Advanced method like contrastive clustering

[Return to top](#Table-of-Contents)  

## Audience and Stakeholders
- Managers / Employees from in-house Test Segment 
- Competitors: e.g. ST Microelectronic

[Return to top](#Table-of-Contents)  