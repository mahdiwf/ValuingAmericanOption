# ValuingAmericanOption
In this case study, I will use Jupyter Notebook to implement the numerical simulation/calculation mentioned on the "section 1 Numerical Example" from Longstaff-Schwartz paper.
Link to the paper is https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf.<br>
This Longstaff-Schwartz algorithm is also known as the Least Squares Monte-Carlo (LSM) algorithm.<br>

The *least squares method* is a form of mathematical regression analysis used to determine the line of best fit for a set of data, providing a visual demonstration of the relationship between the data points. https://www.investopedia.com/terms/l/least-squares-method.asp <br>

By emulating this paper's numerical example in Jupiter Notebook/Python, I hope I will learn interesting world of option for both European & American style from the perspective of programming.<br>

The paper used the following put option as the basis for the simulation: <br>
Consider an American put option on a share of non-dividend-paying stock. The put option is exercisable at a strike price of $1.10 at times 1, 2, and 3, where time 3 (t3) is the final expiration
date of the option. The riskless rate is 6%. The current stock price is $1.00. <br> The objective is to find the stopping rule that maximizes the value of this option.<br>

For better clarity, I will also use the graph, table, and explanation from the paper to explain the steps in this program.<br>

**Price Path** <br>
The paper assumes that the stock price follow the following 8 paths. <br>

![image](https://github.com/user-attachments/assets/33a264d4-89c0-4376-867b-b1e904f47a79) <br>

The steps below will produce the above table.<br>
![image](https://github.com/user-attachments/assets/1ccbf969-d451-4122-9357-61de85ba3f40) <br>
![image](https://github.com/user-attachments/assets/8f779d11-cb28-4355-9de2-9abdaf15e9cd) <br>

Now that we have the potential paths, we need to calculate the cash flows that would be generated if the option was exercised at any point in time. Since it is a put option, if the price is above $1.10, exercising the contract is worthless (or 0).<br>

**Cash Flow** <br>

Since the algorithm is recursive, we first need to compute a number of intermediate matrices. For example, at time 3 (conditional on not exercising the option before the final expiration date), the cash flows realized by the option holder:<br>

![image](https://github.com/user-attachments/assets/2f194d47-6ba3-4087-92bb-5da8ac3dd9ff) <br>
Since I am using DataFrame, this recursive calcalution can be done in one shot for t1, t2, and t3.

![image](https://github.com/user-attachments/assets/cf07eb35-e1c1-40b8-ae25-50dae148d914) <br>

**time 3** <br>
Exercising at t3, produce the European price. Using Numpy's exponential function to discount the cash flows at t3 (future value) to t0 (present value). <br>
![image](https://github.com/user-attachments/assets/1ad71ee2-e6eb-44cd-8c62-2606b065651b) <br>

This value is the same as in the paper.

**time 2** <br>
If the put is in the money at t2, the option holder must then decide whether to exercise the option immediately or continue the option's life until the final expiration date at t3. From the stock-price matrix, there are only five paths for which the option is in the money at t2. Let X denote the stock prices at t2 for these five paths and Y denote the corresponding discounted cash flows received at t3 if the put is not exercised at t2. The vectors X and Y are given by the nondashed entries below.<br>

![image](https://github.com/user-attachments/assets/59fe027c-6198-4efb-872e-c15710bfb51f) <br>

In the steps below, I will produce this table. The 0.9417645336 = np.exp(-r) is used to discount the cash flow at t3 to t2. Where r is risk free rate (6%).
To estimate the conditional expectation value, the paper used "least squares method". For the Python implementation, this is equivalent to the Linear Regression funtion from Scikit-Learn.<br>
![image](https://github.com/user-attachments/assets/bbe2341a-7efd-4dbf-acf1-7c901db788fe) <br>
Path #2, #5 and #8 gone as they are not in_the_money.<br>

With this conditional expectation calculation, we now can compare the value of immediate exercise at t2, given in the first column below, with the value from continuation (to t3), given in the second column below.<br>
![image](https://github.com/user-attachments/assets/10a62306-a126-4539-8b72-845f6a31ed68) <br>

The steps below, I will produce the table above.<br>
![image](https://github.com/user-attachments/assets/153c12ba-2a55-4f2c-9f5b-a0d2f61161cb) <br>
Note:
1) Path #2, #5 and #8 are not in_the_money.<br>
2) calc = Conditional expectation of holding the put calculated using LinearRegression <br>
3) paper = Values from the paper for comparison purpose only <br>
The calculated values are very much similar to the values in the paper.<br>

If we compare the value of column 'exercice' and column 'calc' on the table above, it is optimal to exercise the option at t2 for the 4th, 6th, and 7th paths. This leads to the following matrix, which shows the cash flows received by the option holder conditional on not exercising prior to t2. <br>
![image](https://github.com/user-attachments/assets/44cb6069-ddc1-49f9-9372-7bcca0662c2d) <br>

The steps below, I will produce the table above.<br>
![image](https://github.com/user-attachments/assets/64bf877a-1f46-4254-a2c6-cd4c740d33ea) <br>
When the option is exercised at t2, the cash flow in the final column (t3) becomes zero, because once the option is exercised there are no further cash flow (as option can only be exercised once).<br>

**time 1** <br>
From the stock price matrix, there are again five paths where the option is in the money at t1. Similarly, let X represents the stock price of in-the-money options at t1, and Y is the discounted cash flow at t2.
Since the option can only be exercised once, future cash flows occur at either t2 or t3, but not both. Cash flows received at t2 or t3 are discounted to t1 using Numpy's exponential function.<br>

![image](https://github.com/user-attachments/assets/b5cae9da-ce48-46bc-86dc-f883beb6ddec) <br>

The steps below, I will produce the table above.<br>
![image](https://github.com/user-attachments/assets/d5ca8575-8a91-4224-8ded-3a5d8ddd8018) <br>
Path #2, #3 and #5 gone as they are not in_the_money.<br>

With this conditional expectation calculation, we now can compare the value of immediate exercise at t1, given in the first column below, with the value from continuation (to t2), given in the second column below.<br>
![image](https://github.com/user-attachments/assets/153beda9-41a1-4f5c-b1a7-56a83398562d) <br>

The steps below, I will produce the table above.<br>
![image](https://github.com/user-attachments/assets/17a24c50-8712-4ec3-9558-b69619b43e7f) <br>

Note:
1) Path #2, #3 and #5 are not in_the_money.<br>
2) calc = Conditional expectation of holding the put calculated using LinearRegression <br>
3) paper = Values from the paper for comparison purpose only <br>
The calculated values are very much similar to the values in the paper.<br>
Comparing the two columns shows that exercise at t1 is optimal for the 4th, 6th,7th, and 8th paths. <br>

**stopping rule** <br>
Having identified the exercise strategy at times 1, 2, and 3, the stopping rule can now be represented by the following matrix, where the ones denote exercise dates at which the option is exercised.<br>
![image](https://github.com/user-attachments/assets/51e6d185-ff55-4f0b-ba0b-c9940587f081) <br>

The steps below, I will produce the table above.<br>
![image](https://github.com/user-attachments/assets/ebac3d8f-9fd5-492a-8974-3146986e3d10) <br>

With this specification of the stopping rule, it is now straight forward to determine the cash flows realized by following this stopping rule. This is done by simply exercising the option at the exercise dates where there is a '1' in the above matrix. This leads to the following option cash flow matrix.<br>
![image](https://github.com/user-attachments/assets/28e6b573-eebc-4c8a-8b61-32e5ec7b85c2) <br>
Now, if I print the cash flow matrix: <br>
![image](https://github.com/user-attachments/assets/33213186-9bff-4418-b879-532ddb9121fd)<br>
Having identified the cash flows generated by the American put at each date along each path, the option can now be valued by discounting each cash flow in the option cash flow matrix back to t0, and averaging over all paths.<br>
![image](https://github.com/user-attachments/assets/5a35cc6f-a58b-479d-b948-08db5bb3f146) <br>

The results is a value of **0.1144** for the American put. This is roughly twice the value of **0.0564** for the European put.<br>

**Conclusion**<br>
The American style option is more interesting and complex than the European style because it can be exercised before the exercise date.<br>

References:<br>
https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf.<br>
https://www.investopedia.com/terms/l/least-squares-method.asp <br>
https://medium.datadriveninvestor.com/a-complete-step-by-step-guide-for-pricing-american-option-712c84aa254e <br>
https://medium.com/@ptlabadie/pricing-american-options-in-python-8e357221d2a9 <br>
https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/2.3%20American%20Options.ipynb <br>






