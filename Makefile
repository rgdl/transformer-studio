create:
	aws cloudformation create-stack \
		--stack-name transformer-studio \
		--template-body cloudformation/template.yaml \
		--parameters 'ParameterKey=WebDomainParam,ParameterValue=transformer-studio.com'
