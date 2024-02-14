# Product_Sorting
It includes strategies for sorting products in marketplaces.


# Application : Course Sorting


# Sorting by Rating

# Sorting by Comment Count or Purchase Count

# Sorting by Rating, Comment and Purchase


# Bayesian Average Rating Score

* Bayesian Average Score trims the scores, this is a value derived from the probability method, it is a probabilistic average. <br/>
* Therefore, when this value is used, when a rating calculation is made according to the Bayesian method of your customers (corporate customers), it is open to discussion whether it should be used or not, but it can also be preferred because it will crop the scores of the courses or products a little bit and show them lower. <br/>
* These discussed methods can be added together as weights. <br/>

# Hybrid Sorting: BAR Score + Other Factors

* This solution was ranked in a way that gave a chance to new potential stars with both business knowledge and scientific knowledge.<br/>

* If the result at this stage is interpreted --><br/>
Previously, there was a ranking based on purchases and number of comments, but on the other hand, ratings were also taken into account.<br/>

* Here, a scientific scoring method for ratings was included.<br/>

* The ranking at this stage was also based on wss score, what was the improvement in this part?

* Course_9's high ranking is quite valuable, it has a significant number of reviews and a significant number of purchases.

* Course_1's position is also important, it's a new course with a very low number of reviews and purchases, but a high score, so it might have potential.

* Bar_score gives us the chance to move up those that are new but have potential.

* Ranking is a task that seems very simple from the outside, but when you get into it, there are a lot of bussiness parameters.

* Let's say a person wants to enter a market on a platform or amazon, it is theoretically impossible to enter the market after a certain time because there is market dominance.


** To summarize -->
* When the bar score method is considered as a factor that has weight in a hybrid ranking, it somehow brings up products that have higher potential but have not yet received sufficient social proof.

* The score we obtain probabilistically over 5,4,3,2,1 distributions actually expresses the potential.

For more information, visit my medium article --> https://medium.com/@merveatasoy48/product-sorting-94e9db162789
