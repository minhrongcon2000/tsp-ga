from city import Country
from ga import TSPGA


country = Country(num_cities=10)
learner = TSPGA(country, debug=True)
learner.learn(mode="save", total_timestep=100, filename="result.gif")
