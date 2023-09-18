import asyncio
from typing import Optional

import langchain
from langchain import prompts
from langchain.llms import openai
from langchain.prompts import few_shot as few_shot_prompts
from langchain.schema import language_model

from know_net import base, langchain_utilities, mixins

DEFAULT_MAX_CONCURRENCY = 20

SYSTEM_PROMPT_TEXT = """
You're a seasoned analyst who identifies causal relationships from daily news. Your expertise lies in deriving fact-based connections.

You adhere to a specific schema for causal relationships:
Subject: The cause
Predicate: The event
Object: The effect

Each extracted relationship follows the format:
<Relationship>
  <Subject>{subject}</Subject>
  <Predicate>{predicate}</Predicate>
  <Object>{object}</Object>
</Relationship>
"""
SYSTEM_PROMPT = langchain_utilities.construct_chat_prompt(
    SYSTEM_PROMPT_TEXT, langchain_utilities.Roles.SYSTEM
)

USER_PROMPT_TEMPLATE = """
The following article was extracted from the news article: {{source}}:
Article: {{article}}
Extract the all causal relationships from the article, strictly adhering to the schema.
If there are no relevant causal relationships, respond with "None".
"""
USER_PROMPT = langchain_utilities.construct_chat_prompt(
    USER_PROMPT_TEMPLATE, langchain_utilities.Roles.USER
)

LM_RESPONSE_TEMPLATE = """
{{fewshot_response}}
"""
LM_RESPONSE = langchain_utilities.construct_chat_prompt(
    LM_RESPONSE_TEMPLATE, langchain_utilities.Roles.BOT
)

FEWSHOT_EXAMPLES = [
    {
        "source": """
https://news.yahoo.com/finance/news/fed-decision-fedex-earnings-and-the-uaw-strike-what-to-know-this-week-140006678.html
""",
        "article": """
Jerome Powell and the Federal Reserve will take center stage in the week ahead when the central bank makes its next policy decision.\xa0\nThe Fed is scheduled to meet on Sept. 19 and 20 with the central bank set to release its latest policy decision on Wednesday at 2:00 p.m. ET, followed by a press conference with Powell at 2:30 p.m. ET. Investors expect the FOMC will hold its benchmark interest rate steady in a range of 5.25%-5.5%.\nOn the corporate side, FedEx (FDX)\xa0is expected to report earnings on Wednesday while all eyes will remain on the United Auto Workers strike that began on Friday and is expected to impact production at Stellantis (STLA), GM (GM), and Ford (F).\nMarkets were choppy last week, ending the five-day period mixed\xa0after rising energy prices drove surprises in economic data, but didn\'t significantly change investor bets on interest rates remaining unchanged this week.\nMidway through September, a traditionally tough month for markets, the tech-heavy Nasdaq (^IXIC) has slid 2.3%. The benchmark S&P 500 (^GSPC) is down 1.2%\xa0while the Dow Jones Industrial Average (^DJI) has fallen 0.3%.\nWhen it paused interest rate hikes in July for the first time in 10 meetings, Powell indicated the Fed would remain data dependent. He highlighted several data releases the central bank had eyes on for further insight on the labor market and inflation\'s cooldown.\nThe data since has shown easing core inflation and a cooling labor market, both outcomes the Fed wants. The question surrounding the central bank\'s meeting this week is less about what the Fed does in September and more about the policy decisions later this year.\n"With inflation continuing to run above target and the labor market cooling off only gradually, we expect the Committee to signal further policy tightening is possible if incoming data warrant it," Wells Fargo\'s team of economists wrote in a research note on Friday. "This message is likely to be delivered through both the post-meeting statement and Chair\'s press conference."\nWith the Fed\'s interest rate decision on Wednesday mainly priced in by markets, Bank of America\'s Michael Gapen says the release will be all "about the SEP." The Summary of Economic Projections, which includes Fed officials\' forecasts for inflation, economic growth, and a "dot plot" mapping out expectations for future interest rates, will also be released on Wednesday.\nThe last dot plot released in June showed policymakers projecting an additional rate hike in 2023. Investors will closely watch if that projection for the fed funds rate to peak at 5.6% this year will be moved down, indicating the Fed is done hiking interest rates for the year. Additionally, given the stronger-than-expected economic data over the month of August, investors will be looking for future forecasts on rate cuts.\n"We expect the 2023 median policy rate forecast to show one more 25bp hike, for a terminal rate of 5.5-5.75%," Gapen wrote. "Perhaps the most important forecast is the 2024 median, which in our view will shift up by 25bp to 4.875%, reflecting just 75bp of cuts next year. This would be about 40bp above current market pricing. Risks are skewed toward an even larger upshift in the 2024 median, which would be a significantly hawkish outcome."\nElsewhere, investor attention will remain on the United Auto Workers strike from the Big Three automakers Ford, GM, and Stellantis.\nAfter not reaching a new contract agreement, 13,000 UAW members entered a strike at GM\'s Wentzville, Mo., plant (which assembles midsize trucks and full-size vans), Stellantis\' Toledo Assembly (Jeep Wrangler and Gladiator), and Ford\'s Michigan Assembly Plant in Wayne (Ranger midsize pickup and Bronco SUV).\nThe UAW has threatened to increase the number of striking workers as negotiations go on, which many fear could have repercussions across the broader economy that include\xa0hits to GDP, the labor market and the tech sector. It could even be one of the factors that pushes the Fed to keep rates steady.\n"The impact would cloud the incoming economic data for the next few months, making it harder for the Fed to claim that the figures are breaking decisively one way or the other," Oxford Economics economists Michael Pearce and Nancy Vanden Houten wrote in a note on Wednesday. "That would add, at the margin, to the case for a pause in rates."\nOn Friday afternoon UAW President Shawn Fain released a statement saying they don\'t believe their strike will hurt the economy, just the "billionaire economy."\nBroadly, stocks have been quiet\xa0of late. The CBOE Volatility Index has been hovering around 13, indicating markets have lacked volatility in recent weeks, according to Charles Schwab Investment Strategist Jeffrey Kleintop.\n"We haven\'t seen volatility this low since pre-pandemic period," Kleintop said. "So the market\'s certainly pricing in clear sailing from here, and it may not be that smooth of a ride."\nKleintop is focused on a surge in West Texas Intermediate (CL=F) and Brent crude futures (BZ=F) above $90 a barrel. The energy price increases could send earnings projections lower, per Kleintop, just as they already have for American Airlines (AAL), Spirit (SAVE), and others.\nAirlines are a tiny part of the overall market, but the read across to other consumer businesses and companies with oil as a big cost is important as we move through the next couple of weeks here in this preannouncement season," Kleintop said. "Maybe more confessions could weigh on the market."\nWeekly Calendar\nEconomic data: NAHB housing market index, September (49 expected, 50 previously)\nEarnings: Stitch Fix (SFIX)\nEconomic data: Housing starts, August (1.44 million expected, 1.45 million previously); Housing starts month-over-month (-0.8% expected, +3.9% previously); Building permits, August (1.44 million expected, 1.44 million prior); Building permits month-over-month (-0.2% expected, +0.1% previously)\nEarnings: AutoZone (AZO)\nEconomic data: MBA Mortgage Applications, week ending Sept. 15 (-0.8% previously); FOMC Rate Decision, upper bound (5.5% expected, 5.5% previously); FOMC Rate Decision, lower bound (5.25% expected, 5.25% previously)\nEarnings: FedEx (FDX), General Mills (GIS), KB Home (KBH)\nEconomic data: Initial jobless claims, week ended September 16 (235,000 expected, 208,000 previously); Existing home sales, month-over-month, August (+0.7% expected, -2.2% previously); Leading Index, August (-0.5% expected, -0.4% prior)\nEarnings: FactSet (FDS), Scholastic (SCHL)\nEconomic data: S&P Global US manufacturing PMI, September, preliminary (47.8 expected, 47.9 previously); S&P Global US services PMI, September, preliminary (50.3 expected, 50.5 previously); S&P Global US composite PMI, September, preliminary (50 expected, 50.2 previously)\nEarnings: No notable earnings.\nJosh Schafer is a reporter for Yahoo Finance.\nClick here for the latest economic news and indicators to help inform your investing decisions.\nRead the latest financial and business news from Yahoo Finance\nRelated Quotes'
""",
        "fewshot_response": """
<Relationship>
  <Subject>Rising energy prices</Subject>
  <Predicate>drove surprises</Predicate>
  <Object>economic data</Object>
</Relationship>

<Relationship>
  <Subject>Rising energy prices</Subject>
  <Predicate>didn't significantly change</Predicate>
  <Object>investor bets on interest rates remaining unchanged</Object>
</Relationship>

<Relationship>
  <Subject>United Auto Workers strike</Subject>
  <Predicate>began</Predicate>
  <Object>expected to impact production at Stellantis, GM, and Ford</Object>
</Relationship>

<Relationship>
  <Subject>United Auto Workers strike</Subject>
  <Predicate>continuing</Predicate>
  <Object>could have repercussions across the broader economy including hits to GDP, the labor market, and the tech sector</Object>
</Relationship>

<Relationship>
  <Subject>United Auto Workers strike</Subject>
  <Predicate>continuing</Predicate>
  <Object>could be one of the factors that pushes the Fed to keep rates steady</Object>
</Relationship>

<Relationship>
  <Subject>Easing core inflation and a cooling labor market</Subject>
  <Predicate>have been observed</Predicate>
  <Object>outcomes the Fed wants</Object>
</Relationship>

<Relationship>
  <Subject>Surge in West Texas Intermediate and Brent crude futures above $90 a barrel</Subject>
  <Predicate>could send</Predicate>
  <Object>earnings projections lower for American Airlines, Spirit, and others</Object>
</Relationship>
""",
    },
    # {"source": "2", "article": "2a", "fewshot_response": "2f"},
]
FEWSHOT_TEMPLATE = prompts.ChatPromptTemplate.from_messages([USER_PROMPT, LM_RESPONSE])
FEWSHOTS = few_shot_prompts.FewShotChatMessagePromptTemplate(
    example_prompt=FEWSHOT_TEMPLATE,
    examples=FEWSHOT_EXAMPLES,
)

EXTRACT_TRIPLETS_PROMPT = prompts.ChatPromptTemplate.from_messages(
    [SYSTEM_PROMPT, FEWSHOTS, USER_PROMPT]
)

EXTRACT_TRIPLETS_PROMPT.format(source=content_piece.source, article=content_piece.data)


class CausalityExtractorChain(
    base.AsynchronousTripletExtractor,
    mixins.WithDiskCache,
):
    def __init__(
        self,
        graph_index_creator: language_model.BaseLanguageModel,
        semaphore: Optional[asyncio.Semaphore] = None,
    ) -> None:
        super().__init__()
        self.graph_index_creator = graph_index_creator
        self.semaphore = semaphore or asyncio.Semaphore(DEFAULT_MAX_CONCURRENCY)


if __name__ == "__main__":
    lm = openai.OpenAIChat(model="gpt-4")  # type: ignore

    prompt = langchain.LLMChain()
