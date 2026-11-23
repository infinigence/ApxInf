use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub error_type: &'static str,
    pub param: Option<String>,
    pub code: Option<&'static str>,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>, param: impl Into<Option<String>>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            param: param.into(),
            code: None,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            error_type: "server_error",
            param: None,
            code: Some("model_unavailable"),
        }
    }

    pub fn queue_full() -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: "the batch-one inference queue is full".into(),
            error_type: "rate_limit_error",
            param: None,
            code: Some("queue_full"),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            error_type: "server_error",
            param: None,
            code: None,
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    error_type: &'a str,
    param: Option<&'a str>,
    code: Option<&'a str>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorEnvelope {
            error: ErrorBody {
                message: &self.message,
                error_type: self.error_type,
                param: self.param.as_deref(),
                code: self.code,
            },
        };
        (self.status, Json(body)).into_response()
    }
}
