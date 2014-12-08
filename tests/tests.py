from locust import HttpLocust, TaskSet, task
from pyquery import PyQuery

class WebsiteTasks(TaskSet):
    def on_start(self):
        self.index_page()
        self.urls_on_front_page = self.toc_urls

    @task(15)
    def index_page(self):
        r = self.client.get("/")
        pq = PyQuery(r.content)
        link_elements = pq(".toctree-wrapper a.internal")
        self.toc_urls = [
            l.attrib["href"] for l in link_elements
        ]

class TustUser(HttpLocust):
    task_set = WebsiteTasks
    host = "http://facedetect.aang.in"
    min_wait = 2 *10
    max_wait = 60 *10
