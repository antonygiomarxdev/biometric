/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Body_extract_diagnostic_extract_diagnostic_post } from '../models/Body_extract_diagnostic_extract_diagnostic_post';
import type { Body_extract_minutiae_extract_post } from '../models/Body_extract_minutiae_extract_post';
import type { Body_identify_fingerprint_identify_post } from '../models/Body_identify_fingerprint_identify_post';
import type { Body_register_fingerprint_register_post } from '../models/Body_register_fingerprint_register_post';
import type { DiagnosticResponse } from '../models/DiagnosticResponse';
import type { ExtractResponse } from '../models/ExtractResponse';
import type { HealthResponse } from '../models/HealthResponse';
import type { IdentifyResponse } from '../models/IdentifyResponse';
import type { MetricsResponse } from '../models/MetricsResponse';
import type { RegisterResponse } from '../models/RegisterResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DefaultService {
    /**
     * Health Check
     * Verifica el estado del sistema.
     * @returns HealthResponse Successful Response
     * @throws ApiError
     */
    public static healthCheckGet(): CancelablePromise<HealthResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/',
        });
    }
    /**
     * Extract Minutiae
     * Extrae minutiae de una imagen de huella.
     *
     * Args:
     * file: Archivo de imagen (BMP, PNG, JPEG)
     *
     * Returns:
     * Conteo de minutiae extraídas
     * @param formData
     * @returns ExtractResponse Successful Response
     * @throws ApiError
     */
    public static extractMinutiaeExtractPost(
        formData: Body_extract_minutiae_extract_post,
    ): CancelablePromise<ExtractResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/extract',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Register Fingerprint
     * Registra una nueva huella en el sistema.
     *
     * Args:
     * person_id: ID único de la persona
     * name: Nombre completo
     * document: Número de documento
     * file: Archivo de imagen de la huella
     *
     * Returns:
     * Confirmación de registro
     * @param formData
     * @returns RegisterResponse Successful Response
     * @throws ApiError
     */
    public static registerFingerprintRegisterPost(
        formData: Body_register_fingerprint_register_post,
    ): CancelablePromise<RegisterResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/register',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Identify Fingerprint
     * Identifica una huella buscando coincidencias en el sistema.
     *
     * Args:
     * file: Archivo de imagen de la huella
     *
     * Returns:
     * Resultado de la identificación
     * @param formData
     * @returns IdentifyResponse Successful Response
     * @throws ApiError
     */
    public static identifyFingerprintIdentifyPost(
        formData: Body_identify_fingerprint_identify_post,
    ): CancelablePromise<IdentifyResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/identify',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Fingerprint Image
     * Recupera la imagen original de una huella.
     *
     * Args:
     * person_id: ID de la persona
     *
     * Returns:
     * Imagen (image/bmp)
     * @param personId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getFingerprintImageFingerprintsPersonIdImageGet(
        personId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/fingerprints/{person_id}/image',
            path: {
                'person_id': personId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Fingerprint Details
     * Recupera detalles y minucias guardadas.
     *
     * Args:
     * person_id: ID de la persona
     *
     * Returns:
     * JSON con metadatos y minucias
     * @param personId
     * @returns any Successful Response
     * @throws ApiError
     */
    public static getFingerprintDetailsFingerprintsPersonIdDetailsGet(
        personId: string,
    ): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/fingerprints/{person_id}/details',
            path: {
                'person_id': personId,
            },
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Get Metrics
     * Obtiene métricas de performance del sistema.
     * @returns MetricsResponse Successful Response
     * @throws ApiError
     */
    public static getMetricsMetricsGet(): CancelablePromise<MetricsResponse> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/metrics',
        });
    }
    /**
     * Extract Diagnostic
     * Endpoint de diagnóstico para entender por qué no se extraen minutiae.
     * @param formData
     * @returns DiagnosticResponse Successful Response
     * @throws ApiError
     */
    public static extractDiagnosticExtractDiagnosticPost(
        formData: Body_extract_diagnostic_extract_diagnostic_post,
    ): CancelablePromise<DiagnosticResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/extract/diagnostic',
            formData: formData,
            mediaType: 'multipart/form-data',
            errors: {
                422: `Validation Error`,
            },
        });
    }
    /**
     * Reset Metrics
     * Resetea las métricas acumuladas.
     * @returns any Successful Response
     * @throws ApiError
     */
    public static resetMetricsMetricsResetPost(): CancelablePromise<any> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/metrics/reset',
        });
    }
}
