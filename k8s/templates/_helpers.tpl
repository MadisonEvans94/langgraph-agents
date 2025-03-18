{{- define "agent-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "agent-stack.classifierName" -}}
{{- printf "%s-classifier" (include "agent-stack.name" .) -}}
{{- end -}}

{{- define "agent-stack.agentName" -}}
{{- printf "%s-agent" (include "agent-stack.name" .) -}}
{{- end -}}

{{- define "agent-stack.classifierFullname" -}}
{{- include "agent-stack.classifierName" . -}}
{{- end -}}

{{- define "agent-stack.agentFullname" -}}
{{- include "agent-stack.agentName" . -}}
{{- end -}}
