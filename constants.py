PRETRAINED_SENTENCE_TRANSFORMERS_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

SKILLS_PATTERN_PATH = './data/entity_ruler/skill_patterns.jsonl'
JOB_TITLE_PATH = './data/job_title/Job Title.csv'

DUMMY_JOB_DESCRIPTION = """
As Software Engineer - Backend, you will work in a cross-functional project team to ensure a multitude of systems across products, business, and operations integrate efficiently. You’ll be responsible for designing, building, improving, or maintaining Traveloka services related to new products, business models, business growth, market expansion and process optimization. In addition, you will be Identifying pain points in internal business processes and developing effective, reliable and scalable solutions to work within the expected complexity of the processes. You’ll be expected to deliver the best in class architecture, solution and code. The successful candidate will encounter challenges related to information systems, business process and technology.

As a Backend Software Engineer, you are expected to:

Be responsible for designing, building, improving, or maintaining our backend applications, third-party data integration, data API, backend systems, or working with monitoring tools and infrastructure
Work in cross-functional teams and meet great people regularly from top tier technology, consulting, product, or academic background
Be encouraged to speak your mind, propose ideas, influence others, and continuously grow yourself
Participate and contribute to engineering hygiene such as code review, unit testing, and integration testing
Participate and contribute to the solution and architectural design review.
Participate in the service support as on-call
Participate and contribute to innovation and problem-solving
Post student's scores in Bina Nusantara University's internal application for students Process assistant's honor payment for case making
Schedule important dates for laboratory activities.


‎

Requirements

Bachelor's degree in Computer Science or equivalent from a reputable university with good academic results is preferred.
Having minimum 3 years of experience in software engineering (Java), application development or system development + experience in RDBMS and NoSQL databases.
Experience in version control (Git/SVN/Mercurial) and familiarity with development collaboration tools (GitHub/Phabricator/BitBucket).
Experience in CI/CD like Jenkins/Travis CI/TeamCity and related technologies is a plus.
Experience in AWS/GCP/Azure and other technologies like Ansible, Containers, Kubernetes etc is a plus.
Strong object-oriented analysis and design skills.
Passion in software engineering, application development, or systems development.
Good business acumen, excellent problem skills and broad understanding of software and system design.
Comfortable working up and down the technology stack.
Curiosity to explore creative solutions and try new things to solve challenging problems to pull it all together into a user accepted solution.
Participation in multiple end-to-end implementations of system integration, data migration, internal business applications, or configuring vendor-provided solutions.
Excellent interpersonal, communication, and influence skills and personal maturity.
"""
TEST_TEXT = """michael smith bi / big data/ azure manchester , uk- email me on indeed: indeed.com/r/ falicent/140749dace5dc26f 10+ years of experience in designing, development, administration, analysis, management inthe business intelligence da ta warehousing, client server technologies, web -based applications, cloud solutions and databases. data warehouse: data analysis, star/ snow flake schema data modeling and design specific todata warehousing and business intelligence environment. database: experience in database designing, scalability, back -up and recovery, writing andoptimizing sql code and stored procedures, creating functions, views, triggers and indexes. cloud platform: worked on microsoft azure cloud services like document db, sql azure, streamanalytics, event hub, power bi, web job, web app, power bi, azure data lake analytics(u -sql). big data: worked azure data lake store/analytics for big data processing and azure data factoryto schedule u -sql jobs. designed and developed end to end big data solution for data insights. willing to relocate: anywhere work experience software engineer microsoft - manchester , uk. december 2015 to pre sent 1. microsoft rewards live dashboards: description: - microsoft rewards is loyalty program that rewards users for browsing and shopping online. microsoft rewards members can earn points when searching with bing, browsing with microsoft edge and making purchases at the xbox store, the windows st ore and the microsoft store. plus, user can pick up bonus points for taking daily quizzes and tours on the microsoft rewards website. rewards live dashboards gives a live picture of usage world -wide and by markets like us, canada, australia, new user regis tration count, top/bottom performing rewards offers, orders stats and weekly trends of user activities, orders and new user registrations. the pbi tiles gets refreshed in different frequencies starting from 5 seconds to 30 minutes. technology/tools used event hub, stream analytics and power bi. responsibilities created stream analytics jobs to process event hub data created power bi live dashboard to show live usage traffic, weekly trends, cards, charts to showtop/bottom 10 offers and usage metrics. 2. microsoft rewards data insights: description: - microsoft rewards is loyalty program that rewards users for browsing and shopping online. microsoft rewards members can earn points when searching with bing, browsing with microsoft edge and making purchases at t he xbox store, the windows store and the microsoft store. plus, user can pick up bonus points for taking daily quizzes and tours on the microsoft rewards website. rewards data insights is data analytics and reporting platform, processes 20 million users da ily activities and redemption across different markets like us, canada, australia. technology/tools used cosmos (microsoft big -data platform), c#, x -flow job monitoring, power bi. responsibilities created big data scripts in cosmos c# data extractors, proc essors and reducers for data transformation power bi dashboards 3. end to end tracking tool: description: - this is real -time tracking tool to track different business transactions like order, order response, functional acknowledgement, invoice flowing ins ide icoe. it gives flexibility to customers to track their transactions and appropriate error information in -case of any failure. based on resource based access control the tool gives flexibility to end user to perform different actions like view transacti ons, search based on different filter criteria and view and download actual message payload. end to end tracking tool stitches all the business transaction like order to cash flow and connects different hops inside icoe like gateway, routing server, proces sing server. it also connects different systems like icoe, partner end point and sap. technology/tools used azure document db, azure web job and web app, rbac, angular js. responsibilities document db stored procedures. web job to process event hub data and populate document db• web app api. stream analytics job to transform data power bi reports 4. biztrack tracking tool: description: - this is real -time tracking tool to track different business transactions like order, order response, functional acknowledgement, invoice flowing inside icoe. it gives flexibility to customers to track their transactions and appropriate error information in -case of any failure. based on resource based access control the tool gives flexibility to end user to perform different actions like view transactions, search based on different filter criteria and view and download actual message payload. technology/tools used sql server 2014, ssis, .net api, angular js. responsibilities etl solution to transform business transactions data stored in biztalk tables. sql azure tables, stored procedures, user defined functions. performance tuning. web api enha ncements. education the university of manchester - uk 2007 skills problem solving (less than 1 year), project lifecycle (less than 1 year), project manager (less than 1 year), technical assistance. (less than 1 year) additional information professiona l skills excellent analytical, problem solving, communication, knowledge transfer and interpersonalskills with ability to interact with individuals at all the levels quick learner and maintains cordial relationship with project manager and team members and good performer both in team and independent job environments positive attitude towards superiors &amp; peers supervised junior developers throughout project lifecycle and provided technical assistance.
"""

CATEGORIES_PATTERN = {
    
    "EXPERIENCES": r"\b(create|participate|develop|lead|manage|implement|coordinate|analyze|design|oversee|improve|collaborate|achieve|increase|decrease|reduce|enhance|optimize|maintain|support|train|mentor|supervise|negotiate|present|research|execute|plan|organize|launch|initiate|establish|conduct|evaluate|troubleshoot|resolve|streamline|automate|innovate|pioneer)\b",
    "EDUCATIONS":  r"(?i)(?:Bsc|\bB\.\w+|\bM\.\w+|\bPh\.D\.\w+|\bBachelor(?:'s)?|\bMaster(?:'s)?|\bPh\.D)\s(?:\w+\s){0,5}\w+",
    "YEARS_EXPERIENCES": r"(\d+)\s*(?:[-+]|plus)?\s*years?\b"
    
}

RATING_WEIGHTS = {

    'EDUCATIONS': 0.10,
    'GPA': 0.00, # TO BE UPDATED
    'JOB_TITLES' : 0.20,
    'YEARS_EXPERIENCES': 0.15,
    'EXPERIENCES': 0.075,
    'SKILLS': 0.225,
    'SOFT_SKILLS': 0.10,
    'LANGUAGES': 0.15
        
}

YEARS_EXPERIENCES_WEIGHTS = {
    
    'KEYWORDS_MATCH' : 0.5,
    'KEYWORDS_CONTEXT' : 0.5
    
}
