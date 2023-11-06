template := file://cloudformation/template.yaml
stack_name := transformer-studio

deploy-stack:
	aws cloudformation update-stack \
		--stack-name $(stack_name) \
		--template-body $(template) \
		--parameters \
		'ParameterKey=WebDomainParam,ParameterValue=transformer-studio.com' \
		'ParameterKey=RegionParam,ParameterValue=ap-southeast-2' \
		'ParameterKey=CertificateArnParam,ParameterValue=arn:aws:acm:us-east-1:587086740654:certificate/9fdaf05e-2f75-4ee4-a6b5-191c6dafe2f3' \
		'ParameterKey=MXRecordParam,ParameterValue="10 mx.hover.com.cust.hostedemail.com"'

deploy-site:
	aws s3 sync site-content s3://transformer-studio.com --delete
	aws cloudfront create-invalidation --distribution-id E3N36U9IZEKQNP --paths "/*"

