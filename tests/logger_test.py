# check_clearml.py
from clearml import Task
task = Task.init(project_name="my project", task_name="my task")
task.get_logger().report_single_value("test_val", 1.0)
print("Task URL:", task.get_output_log_web_page())
task.close()