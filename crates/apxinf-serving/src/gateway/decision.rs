use super::*;

pub(super) async fn systemone(State(service): State<Arc<Service>>, body: Bytes) -> Response {
    handle_decision(service, body)
        .await
        .unwrap_or_else(IntoResponse::into_response)
}

async fn handle_decision(service: Arc<Service>, body: Bytes) -> Result<Response, ApiError> {
    if !service.decision_enabled() {
        return Err(ApiError::new(
            400,
            "unsupported_feature",
            "The selected provider does not support decisions.",
        ));
    }
    let request = crate::decision::parse_decision_request(&body)
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    let record = request.record.clone();
    let ticket = service.enqueue_decision(request)?;
    let _guard = CancelGuard(ticket.cancel);
    let begin = ticket.begin.await.map_err(|_| {
        ApiError::new(
            503,
            "worker_lost",
            "The decision worker stopped before admission.",
        )
    })??;
    let mut output = ticket.output;
    match output.recv().await {
        Some(ModelEvent::Decision(result)) => {
            let mut headers = HeaderMap::new();
            headers.insert("x-request-id", begin.request_id.parse().unwrap());
            headers.insert(
                "x-apxinf-model-revision",
                service.ready["model_revision"]
                    .as_str()
                    .unwrap()
                    .parse()
                    .unwrap(),
            );
            headers.insert(
                "x-apxinf-profile",
                public_profile(&service).parse().unwrap(),
            );
            Ok((
                headers,
                Json(crate::decision::project_response(
                    &service.model_id,
                    &record,
                    &result,
                )),
            )
                .into_response())
        }
        Some(ModelEvent::Error(error)) => Err(error),
        _ => Err(ApiError::new(
            500,
            "protocol_fault",
            "The decision attempt ended without a settled decision result.",
        )),
    }
}
