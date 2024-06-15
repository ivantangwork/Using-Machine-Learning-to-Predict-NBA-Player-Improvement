Using machine learning to predict NBA player improvement and future performances

# Project Overview

I have always been a big fan of NBA and I always liked to look at NBA player stats and development of my favourite NBA players over the years. 
It is very important for NBA team management to access how well their players are developing for them to build a team to ultimately win a championship.
For my project, I hope to ultimately ultilise machine learning with python tools, to observe the logic behind how numbers can reflect the development and improvement of NBA players, and hope to predict the trend in the future. 

# Data

The dataset I found is from Kaggle where the author did webscraping from the basketball-reference.com. This website is reowned for storing all the NBA stats.
The dataset contains data from 1995 to 2022/2023 (Last NBA season).
I data scraped an extra 13 years of NBA data from basketball reference website for extra data for my models. Therefore my dataset includes players from 1981 to 2024.

Google drive to all the data files I used for my project: https://drive.google.com/drive/folders/1sEucHQVqPHBsqoGyMpak_m_DXb8exN2P?usp=sharing

# Data Science Problem

Using machine learning to predict NBA player improvement and future performances

To predict NBA player improvement, we need to define the metric that measures a player's improvement.
We will be mainly measuring the target feature "PER Improvement", which is a feature I created using the difference between a player's current season PER and next season's PER. 

The player efficiency rating (PER) is John Hollinger's all-in-one basketball rating, which attempts to collect or boil down all of a player's contributions into one number. Using a detailed formula, Hollinger developed a system that rates every player's statistical performance. PER strives to measure a player's per-minute performance, while adjusting for pace. A league-average PER is always 15.00, which permits comparisons of player performance across seasons. PER takes into account positive results, including field goals, free throws, 3-pointers, assists, rebounds, blocks and steals and negative results, including missed shots, turnovers and personal fouls. The formula adds positive stats and subtracts negative ones through a statistical point value system. The rating for each player is then adjusted to a per-minute basis so that, for example, substitutes can be compared with starters in playing time debates. It is also adjusted for the team's pace. In the end, one number sums up the players' statistical accomplishments for that season. - Wikipedia (https://en.wikipedia.org/wiki/Player_efficiency_rating)

# My Approach

In my project I primarily use regression models for improvement prediction. I am comparing XGBoost Regression, SVM regression, RandomForest Regression and Linear regression, and hope to find the best model for our goal

I have performed several EDAs around PER Improvement and looked at effects of a few hypothesis. Including: Will a star player improve less than a rookie player? Does aging cause decline in performance or restrict improvement? How different performance levels of players affect their improvement in coming season? Player Improvement trend... etc.


# !!! Important !!!

I have made 7 different notebooks, please read in the order: Data Prep and Preprocess > webscraping > EDA > LR & RFR modelling > Xgboost modelling > SVM modelling > Model Evaluations and Comparisons


# Data dictionary:

average_df

| Column name | Datatype | Description                                                                                   |
|-------------|----------|-----------------------------------------------------------------------------------------------|
| Player      | object   | First and last name of the NBA player                                                         |
| Age         | int64    | Age of the player during that season                                                          |
| G           | int64    | The number of games the player had played that season                                         |
| GS          | int64    | The number of games the player played as a starter                                            |
| MP          | float64  | The average minutes the player had played per game                                            |
| FG          | float64  | Field goals - The average number of times the player scored a basket per game                 |
| FGA         | float64  | Field goals attempted - The average number of times the player attempted to score a basket    |
| FG%         | float64  | The percentage of field goals made per game                                                   |
| 3P          | float64  | The number of 3-pointers made per game                                                        |
| 3PA         | float64  | The number of 3-pointers attempts made per game                                               |
| 3P%         | float64  | The percentage of 3-pointers made per game                                                    |
| 2P          | float64  | The number of 2-pointers made per game                                                        |
| 2PA         | float64  | The number of 2-pointers attempts made per game                                               |
| 2P%         | float64  | The percentage of 2-pointers made per game                                                    |
| eFG%        | float64  | Effective Field Goal Percentage; (FG + 0.5 * 3P)/FGA. Adjusts for 3-pointer being one more point than a 2-pointer |
| FT          | float64  | The number of free-throws made per game                                                       |
| FTA         | float64  | The number of free-throws attempts per game                                                   |
| FT%         | float64  | The percentage of free-throws made per game                                                   |
| ORB         | float64  | Offensive rebounds per game - When a player grabs a rebound from the opponents' basket        |
| DRB         | float64  | Defensive rebounds per game - When a player grabs a rebound from his own basket               |
| TRB         | float64  | Total rebounds - The sum of offensive and defensive rebounds the player grabbed per game      |
| AST         | float64  | Assists per game                                                                              |
| STL         | float64  | Steals per game                                                                               |
| BLK         | float64  | Blocks per game                                                                               |
| TOV         | float64  | Turnover per game - A turnover is when a player loses the ball to the opponent                |
| PF          | float64  | Personal fouls per game                                                                       |
| PTS         | float64  | Points per game                                                                               |
| team        | object   | The team name in abbreviation                                                                 |
| season      | object   | The year of that season started / the year of that season ended                               |
| team_retcon | object   | Team abbreviation                                                                             |


roster_df

| Column name | Datatype | Description                                                      |
|-------------|----------|------------------------------------------------------------------|
| Player      | object   | First and last name of the NBA player                            |
| Pos         | object   | Position of the player                                           |
| Ht          | object   | Height of the player in feet and inches                          |
| Wt          | int64    | Weight of the player in pounds                                   |
| Birth Date  | object   | Date of birth of the NBA player (Month dd, yyyy)                 |
| Unnamed: 6  | object   | The country the player is from                                   |
| Exp         | int64    | The number of years the player has played in the NBA             |
| College     | object   | The college of the U.S. players before joining the NBA           |
| team        | object   | The players' team name in abbreviation                           |
| season      | object   | The year of that season started / the year of that season ended  |
| team_retcon | object   | The players' team name in abbreviation



advanced_df

| Column name | Datatype | Description                                                                                                                                                                                      |
|-------------|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MP          | int64      | Total minutes played by a player in a season.                                                                                                                                                    |
| PER         | float    | Player Efficiency Rating - The PER sums up all a player's positive accomplishments, subtracts the negative accomplishments, and returns a per-minute rating of a player's performance.           |
| TS%         | float    | True Shooting Percentage - a measure of shooting efficiency that takes into account field goals, 3-point field goals, and free throws.                                                           |
| 3PAr        | float    | 3-Point Attempt Rate- Percentage of total FG Attempts that are from 3-Point Range.                                                                                                               |
| FTr         | float    | Free Throw Rate - Measures Free Throws per field goal attempt.                                                                                                                                   |
| ORB%        | float    | Offensive Rebound Percentage - an estimate of the percentage of available offensive rebounds a player grabbed while he was on the floor.                                                         |
| DRB%        | float    | Defensive Rebound Percentage - an estimate of the percentage of available defensive rebounds a player grabbed while he was on the floor.                                                         |
| TRB%        | float    | Total Rebound Percentage - an estimate of the percentage of available rebounds a player grabbed while he was on the floor.                                                                       |
| AST%        | float    | Assist Percentage - The percentage of teammate field goals a player assisted on while they were on the floor                                                                                     |
| STL%        | float    | The ratio of a player's steals to the total number of opponent possessions expressed as a percentage.                                                                                            |
| BLK%        | float    | The percentage of opponent shots that a player blocks.                                                                                                                                           |
| TOV%        | float    | A player's TOV to his own team's possessions expressed as a percentage.                                                                                                                          |
| USG%        | float    | Usage Rate - an estimate of the percentage of a team's offensive possessions used by an individual player during his time on the floor.                                                          |
| OWS         | float    | Offensive Win Shares                                                                                                                                                                             |
| DWS         | float    | Defensive Win Shares                                                                                                                                                                             |
| WS          | float    | Winshares - a player statistic which attempts to divvy up credit for team success to the individuals on the team.                                                                                |
| WS/48       | float    | Win Shares Per 48 Minutes - an estimate of the number of wins contributed by the player per 48 minutes (league average is approximately 0.100).                                                  |
| OBPM        | float    | offensive box plus/minus                                                                                                                                                                         |
| DBPM        | float    | Defensive box plus/minus                                                                                                                                                                         |
| BPM         | float    | Box Plus/Minus - a basketball box score-based metric that estimates a basketball player's contribution to the team when that player is on the court.                                             |
| VORP        | float    | Value Over Replacement Player - a statistic used in basketball to measure a player's overall contributions to their team, when compared to an average "replacement" player at the same position. |



