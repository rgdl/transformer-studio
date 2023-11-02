template := file://cloudformation/template.yaml
stack_name := transformer-studio

deploy:
	aws cloudformation update-stack \
		--stack-name $(stack_name) \
		--template-body $(template) \
		--parameters 'ParameterKey=WebDomainParam,ParameterValue=transformer-studio.com'

validate:
	aws cloudformation validate-template \
		--template-body $(template)
