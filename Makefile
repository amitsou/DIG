install:
	poetry install

shell:
	poetry shell

update: # Update dependencies (lockfile and installed packages)
	poetry update

outdated: # Check for outdated dependencies
	poetry show --outdated

show:
	poetry show --tree

clean:
	poetry env remove $(shell poetry env list --full-path | awk '{print $$1}')
	rm -rf __pycache__
	rm -rf .pytest_cache
