from src.replay import ExperienceBuffer
buf = ExperienceBuffer(capacity=10)
print(isinstance(buf, ExperienceBuffer))