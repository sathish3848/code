from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai import LLM

# Specify the Gemini model
# Specify the Gemini model
gemini_llm = LLM(model="gemini/gemini-3.8-flash")


@CrewBase
class EnterpriseMarketIntelCrew:
    """EnterpriseMarketIntel crew setup"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"], llm=gemini_llm, verbose=True
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(config=self.agents_config["analyst"], llm=gemini_llm, verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the EnterpriseMarketIntel crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
