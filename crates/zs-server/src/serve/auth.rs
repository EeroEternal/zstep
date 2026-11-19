// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

use super::*;

pub(super) fn constant_time_eq(provided: &[u8], expected: &[u8]) -> bool {
    let provided = std::hint::black_box(provided);
    let expected = std::hint::black_box(expected);
    let mut diff = std::hint::black_box(provided.len() ^ expected.len());
    for (idx, &exp_byte) in expected.iter().enumerate() {
        let prov_byte = provided.get(idx).copied().unwrap_or(0);
        diff = std::hint::black_box(diff | ((prov_byte ^ exp_byte) as usize));
    }
    std::hint::black_box(diff) == 0
}

pub(super) fn extract_bearer_token(headers: &HeaderMap) -> Option<&[u8]> {
    let raw = headers.get(header::AUTHORIZATION)?.as_bytes();
    if raw.len() <= 7 || !raw[..7].eq_ignore_ascii_case(b"bearer ") {
        return None;
    }
    let token = &raw[7..];
    let start = token.iter().position(|b| !b.is_ascii_whitespace())?;
    let end = token.iter().rposition(|b| !b.is_ascii_whitespace())?;
    Some(&token[start..=end])
}

pub(super) fn unauthorized_response() -> Response {
    let mut resp = error_response(
        StatusCode::UNAUTHORIZED,
        "authentication_error",
        "invalid_api_key",
        "missing or invalid Authorization Bearer token",
    );
    resp.headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    resp
}

pub(super) fn authorize_bearer(headers: &HeaderMap, state: &AppState) -> Option<Response> {
    let expected = state.api_key.as_ref()?;
    match extract_bearer_token(headers) {
        Some(provided) if constant_time_eq(provided, expected.as_bytes()) => None,
        _ => Some(unauthorized_response()),
    }
}

pub(super) fn is_loopback_peer(peer: Option<SocketAddr>) -> bool {
    peer.is_some_and(|addr| match addr.ip() {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    })
}

pub(super) fn authorize_ops_request(
    req: &Request<Body>,
    state: &AppState,
    allow_remote_unauth: bool,
) -> Option<Response> {
    if state.api_key.is_some() {
        return authorize_bearer(req.headers(), state);
    }
    if allow_remote_unauth {
        return None;
    }
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0);
    if is_loopback_peer(peer) {
        None
    } else {
        Some(error_response(
            StatusCode::FORBIDDEN,
            "permission_error",
            "loopback_only",
            "admin and metrics endpoints require --api-key, --admin-port, or loopback access",
        ))
    }
}
