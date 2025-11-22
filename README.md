# OCR
Integrated Demand Forecasting and Inventory Replenishment in Omni-Channel Retail with Deep Learning

The rise of omni-channel grocery retail presents new challenges for efficient inventory replenishment. This study formulates a novel problem setting in which retailers must jointly manage inventory across the front store (serving offline demand) and the backroom (serving online demand), while allowing flexible but costly inventory sharing between the two. To address this complexity, we develop and compare two solution frameworks for daily, SKU-level replenishment under feature-driven demand: a traditional Separated Forecasting–Optimization (SFO) approach and an End-to-End Deep Learning (E2E) model. Using large-scale real-world data, our empirical results show that the E2E framework consistently achieves lower total costs and better demand–supply balance than two-stage methods across varying cost structures, demand volatility, and product characteristics. The findings underscore not only the importance of modeling inventory sharing costs, but also the value of integrated learning in capturing complex cross-channel dependencies and enhancing decision robustness in omni-channel retail operations.

The codes shared here is used to check the details of the algorithm implementation. To run the experiments with one model: 

sh OCR_E2E_Main.sh
