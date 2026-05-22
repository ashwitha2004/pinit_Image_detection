import { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Upload, Camera, Search, CheckCircle, AlertCircle, Shield, TrendingUp, AlertTriangle, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { extractAdvancedWatermark, type AdvancedWatermarkMetadata } from "@/lib/advancedSteganography";
import { extractSimpleWatermark, type SimpleWatermarkMetadata } from "@/lib/simpleSteganography";
import { analyzeImage, type ImageAnalysisResult } from "@/lib/imageAnalysis";
import { getAIDetection, type IntegratedDetectionResult } from "@/utils/aiDetection/aiDetectionIntegration";
import { CameraCapture } from "@/components/CameraCapture";

interface VerificationResult {
  success: boolean;
  isAuthentic: boolean;
  confidence: number;
  watermarkDetected: boolean;
  pinitEncrypted: boolean;
  imageType: 'camera' | 'ai' | 'screenshot' | 'edited' | 'unknown';
  aiGeneratedProbability: number;
  metadataStatus: 'original' | 'modified';
  compressionDetected: boolean;
  trustScore: number;
  metadata?: AdvancedWatermarkMetadata | SimpleWatermarkMetadata;
  analysis?: ImageAnalysisResult;
  error?: string;
  details: {
    fileName: string;
    timestamp: string;
    detectionType: string;
    issues: string[];
  };
  forensicType: string;
  aiProbability: number;
  cameraProbability: number;
  screenshotProbability: number;
  editedProbability: number;
  downloadedProbability: number;
  detectionType?: string;
  riskLevel?: string;
  debug?: any;
  securityStatus?: string;
  cameraCaptured?: boolean;
  forensicSubScores?: {
    aiTextureScore: number;
    aiEdgeScore: number;
    aiFrequencyScore: number;
    aiSymmetryScore: number;
    aiNoiseScore: number;
    cfaScore: number;
    sensorNoiseScore: number;
    jpegConsistency: number;
    aberrationScore: number;
    edgeRandomness: number;
  };
  suppressionTriggered?: boolean;
}

const VerifyProof = () => {
  const navigate = useNavigate();
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [verificationResult, setVerificationResult] = useState<VerificationResult | null>(null);
  const [verificationMode, setVerificationMode] = useState<'auto' | 'advanced' | 'simple'>('auto');
  const [showCameraModal, setShowCameraModal] = useState(false);

  // ── Failure-reporting state ────────────────────────────────────────────────
  type ReportState = 'idle' | 'selecting' | 'submitting' | 'success' | 'error';
  const [reportState, setReportState]     = useState<ReportState>('idle');
  const [reportMessage, setReportMessage] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Removed AI detection initialization as it's not being used

  const handleImageSelect = (imageData: string, fileName: string, source: 'upload' | 'camera' = 'upload') => {
    setSelectedImage(imageData);
    setSelectedFileName(fileName);
    setVerificationResult(null);
    localStorage.setItem('current_image_source', source);
  };

  const handleCameraCapture = () => {
    setShowCameraModal(true);
  };

  const handleCameraCaptureComplete = (imageData: string) => {
    const fileName = `camera_capture_${Date.now()}.jpg`;
    handleImageSelect(imageData, fileName, 'camera');
    setShowCameraModal(false);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (e.target?.result) {
          handleImageSelect(e.target.result as string, file.name, 'upload');
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const processVerification = async () => {
    if (!selectedImage) return;

    setIsProcessing(true);
    setVerificationResult(null);

    try {
      let result: VerificationResult;
      const fileName = `verified_image_${Date.now()}.jpg`;
      const timestamp = new Date().toISOString();

      // Use existing verification logic
      let watermarkMetadata: AdvancedWatermarkMetadata | SimpleWatermarkMetadata | null = null;
      let watermarkDetected = false;
      let pinitEncrypted = false;
      let detectionType = 'Unknown';

      if (verificationMode === 'auto' || verificationMode === 'advanced') {
        try {
          watermarkMetadata = await extractAdvancedWatermark(selectedImage);
          if (watermarkMetadata) {
            watermarkDetected = true;
            pinitEncrypted = watermarkMetadata.pinitEncrypted || false;
            detectionType = 'Advanced Watermark';
          }
        } catch (e) {
          console.log('Advanced watermark extraction failed, trying simple...');
        }
      }

      if (!watermarkDetected && (verificationMode === 'auto' || verificationMode === 'simple')) {
        try {
          watermarkMetadata = await extractSimpleWatermark(selectedImage);
          if (watermarkMetadata) {
            watermarkDetected = true;
            pinitEncrypted = watermarkMetadata.pinitEncrypted || false;
            detectionType = 'Simple Watermark';
          }
        } catch (e) {
          console.log('Simple watermark extraction failed');
        }
      }

      // Get capture source from localStorage
      const captureSource = localStorage.getItem('current_image_source') as 'upload' | 'camera' | null;
      localStorage.removeItem('current_image_source');
      
      // Perform comprehensive forensic analysis
      const analysis = await analyzeImage(selectedImage, selectedFileName || undefined, captureSource || undefined);
      
      // Use AI detection system
      let aiDetectionResult: IntegratedDetectionResult | null = null;
      let aiProbability = 0;
      
      try {
        const aiDetector = getAIDetection();
        aiDetectionResult = await aiDetector.analyzeImage(selectedImage);
        aiProbability = aiDetectionResult.confidence * 100;
      } catch (error) {
        console.warn('AI detection failed, falling back to forensic analysis:', error);
        aiProbability = analysis.forensicReport?.aiProbability || 0;
      }
      
      const metadataStatus = analysis.metadata.hasExif ? 'original' : 'modified';

      // Phase 1: compressionDetected must reflect *compression*, not editing.
      // The previous wiring (editedProbability > 50) conflated two unrelated
      // detectors. Until the real WhatsApp/Downloaded detectors land (Phase 3),
      // we use a conservative bytes-per-pixel heuristic on JPEG inputs only —
      // a recompressed/downloaded JPEG typically drops well below 0.25 B/pixel,
      // while a fresh camera JPEG sits between ~0.5 and ~3 B/pixel. PNGs and
      // unknown mime types are deliberately excluded (no new false positives).
      let compressionDetected = false;
      try {
        const dimsStr = (analysis.metadata?.dimensions || '').toString();
        const [wStr, hStr] = dimsStr.split('x');
        const w = parseInt(wStr, 10) || 0;
        const h = parseInt(hStr, 10) || 0;
        const pixelCount = w * h;
        // base64 length math: 3 bytes per 4 chars (ignoring padding) — close enough
        const fileBytes = Math.floor((selectedImage.length * 3) / 4);
        const mime = (analysis.metadata?.mimeType || '').toLowerCase();
        const isJpeg = mime.includes('jpeg') || mime.includes('jpg');
        if (isJpeg && pixelCount > 0) {
          const bpp = fileBytes / pixelCount;
          if (bpp < 0.25) compressionDetected = true;
        }
      } catch (e) {
        console.warn('[verify] compression heuristic failed (non-fatal):', e);
      }

      // Phase 1: real trust score (was hard-coded to 100). Weights are aligned
      // with backend/routers/pinit_verification.py so frontend and backend
      // scores stay coherent.
      //   Watermark present .......... 30
      //   PINIT-encrypted ............ 30
      //   Classification confidence .. up to 20 (forensic confidence × 0.2)
      //   EXIF integrity ............. 10
      //   No recompression ........... 10
      let trustScore = 0;
      if (watermarkDetected) trustScore += 30;
      if (pinitEncrypted) trustScore += 30;
      trustScore += Math.round((analysis.confidence || 0) * 0.2);
      if (analysis.metadata.hasExif) trustScore += 10;
      if (!compressionDetected) trustScore += 10;
      trustScore = Math.max(0, Math.min(100, trustScore));

      // Determine authenticity (threshold unchanged from prior behavior)
      const isAuthentic = watermarkDetected && pinitEncrypted && trustScore >= 70;
      
      // Use forensic confidence instead of random values
      const confidence = Math.round(analysis.confidence);

      // Identify issues
      const issues: string[] = [];
      if (!watermarkDetected) {
        issues.push('No PINIT watermark detected');
      }
      if (watermarkDetected && !pinitEncrypted) {
        issues.push('Watermark found but not PINIT encrypted');
      }
      
      if (aiDetectionResult && aiDetectionResult.aiGenerated) {
        issues.push(`AI-generated content detected (${aiProbability.toFixed(1)}% confidence)`);
      } else if (aiProbability > 50) {
        issues.push(`AI-generated content detected (${aiProbability.toFixed(1)}% probability)`);
      }
      
      if (compressionDetected) {
        issues.push('Image compression detected');
      }

      // Map image type to correct enum values
      const imageType = analysis.imageType;
      
      const cameraCaptured = imageType === 'camera';
      const securityStatus = cameraCaptured ? 'Authentic Camera Capture' :
                           imageType === 'ai' ? 'Synthetic AI Generated Image' :
                           imageType === 'screenshot' ? 'Screen Captured Content' :
                           imageType === 'edited' ? 'Manipulated or Edited Image' :
                           'Unable To Verify';

      result = {
        success: true,
        isAuthentic,
        confidence,
        watermarkDetected,
        pinitEncrypted,
        imageType,
        cameraCaptured,
        securityStatus,
        aiGeneratedProbability: aiProbability,
        metadataStatus,
        compressionDetected,
        trustScore,
        metadata: watermarkMetadata || undefined,
        analysis,
        details: {
          fileName,
          timestamp,
          detectionType,
          issues
        },
        // Add missing properties as requested
        forensicType: analysis.imageType,
        aiProbability: analysis.forensicReport?.aiProbability || 0,
        cameraProbability: analysis.forensicReport?.cameraProbability || 0,
        screenshotProbability: analysis.forensicReport?.screenshotProbability || 0,
        editedProbability: analysis.forensicReport?.editedProbability || 0,
        downloadedProbability: analysis.forensicReport?.downloadedProbability || 0,
        suppressionTriggered: analysis.forensicReport?.suppressionTriggered || false,
        forensicSubScores: analysis.forensicReport ? {
          aiTextureScore: analysis.forensicReport.aiSubScores?.textureScore ?? 0,
          aiEdgeScore: analysis.forensicReport.aiSubScores?.edgeScore ?? 0,
          aiFrequencyScore: analysis.forensicReport.aiSubScores?.frequencyScore ?? 0,
          aiSymmetryScore: analysis.forensicReport.aiSubScores?.symmetryScore ?? 0,
          aiNoiseScore: analysis.forensicReport.aiSubScores?.noiseScore ?? 0,
          cfaScore: analysis.forensicReport.cameraSubScores?.cfaScore ?? 0,
          sensorNoiseScore: analysis.forensicReport.cameraSubScores?.sensorNoiseScore ?? 0,
          jpegConsistency: analysis.forensicReport.cameraSubScores?.jpegConsistency ?? 0,
          aberrationScore: analysis.forensicReport.cameraSubScores?.aberrationScore ?? 0,
          edgeRandomness: analysis.forensicReport.cameraSubScores?.edgeRandomness ?? 0,
        } : undefined
      };

      setVerificationResult(result);

    } catch (error) {
      console.error('Verification failed:', error);
      setVerificationResult({
        success: false,
        isAuthentic: false,
        confidence: 0,
        watermarkDetected: false,
        pinitEncrypted: false,
        imageType: 'unknown',
        cameraCaptured: false,
        securityStatus: 'Unable To Verify',
        aiGeneratedProbability: 0,
        metadataStatus: 'modified',
        compressionDetected: false,
        trustScore: 0,
        forensicType: 'unknown',
        aiProbability: 0,
        cameraProbability: 0,
        screenshotProbability: 0,
        editedProbability: 0,
        downloadedProbability: 0,
        suppressionTriggered: false,
        forensicSubScores: undefined,
        error: error instanceof Error ? error.message : 'Verification failed',
        details: {
          fileName: `error_image_${Date.now()}.jpg`,
          timestamp: new Date().toISOString(),
          detectionType: 'Error',
          issues: ['Verification process failed']
        }
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const resetForm = () => {
    setSelectedImage(null);
    setSelectedFileName(null);
    setVerificationResult(null);
    setReportState('idle');
    setReportMessage('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // ── Report-wrong-detection handler ─────────────────────────────────────────
  const reportWrongDetection = async (correction: 'ai' | 'camera' | null) => {
    if (!selectedImage || !verificationResult) return;

    setReportState('submitting');

    // Derive predicted label from the result's AI probability
    const predictedLabel: 'ai' | 'camera' =
      (verificationResult.aiProbability ?? verificationResult.aiGeneratedProbability ?? 0) >= 50
        ? 'ai'
        : 'camera';

    const BACKEND_URL: string =
      (import.meta as any).env?.VITE_BACKEND_URL ||
      (window.location.hostname === 'localhost' ? 'http://localhost:8000' : '');

    try {
      const res = await fetch(`${BACKEND_URL}/api/inference/report-failure`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64:      selectedImage,
          filename:          selectedFileName || 'image.jpg',
          predicted_label:   predictedLabel,
          ai_probability:    (verificationResult.aiProbability ?? 0) / 100,
          camera_probability:(verificationResult.cameraProbability ?? 0) / 100,
          confidence:        verificationResult.confidence ?? 0,
          correction_label:  correction,
        }),
      });

      if (res.ok) {
        setReportState('success');
        setReportMessage('Saved! This image will help improve future accuracy.');
      } else {
        const text = await res.text().catch(() => '');
        console.warn('[ReportFailure] Backend error:', res.status, text.slice(0, 200));
        setReportState('error');
        setReportMessage('Could not save report — please try again later.');
      }
    } catch (err) {
      console.warn('[ReportFailure] Fetch failed:', err);
      setReportState('error');
      setReportMessage('Backend unreachable — report could not be saved.');
    }
  };

  // Helper functions for improved UI display
  const getSecurityStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'authentic':
        return 'text-green-400';
      case 'synthetic media':
        return 'text-red-400';
      case 'digital capture':
        return 'text-yellow-400';
      case 'modified':
        return 'text-orange-400';
      case 'external source':
        return 'text-blue-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low':
        return 'text-green-400';
      case 'medium':
        return 'text-yellow-400';
      case 'high':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-400';
    if (confidence >= 80) return 'text-yellow-400';
    if (confidence >= 70) return 'text-orange-400';
    return 'text-red-400';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sticky top-0 z-40 bg-background/80 backdrop-blur-xl border-b border-border/50 px-4 py-4"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button
              onClick={() => navigate("/home")}
              variant="ghost"
              size="sm"
              className="text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
            <div>
              <h1 className="text-lg font-semibold text-foreground">Verify Proof</h1>
              <p className="text-xs text-muted-foreground">Improved image verification system</p>
            </div>
          </div>
          <Search className="w-6 h-6 text-muted-foreground" />
        </div>
      </motion.div>

      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {!verificationResult ? (
          <>
            {/* Verification Mode Selection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-8"
            >
              <h2 className="text-xl font-semibold mb-4 text-white">Verification Mode</h2>
              <div className="grid md:grid-cols-3 gap-4">
                <button
                  onClick={() => setVerificationMode('auto')}
                  className={`p-4 rounded-lg border transition-all ${
                    verificationMode === 'auto'
                      ? 'bg-purple-600/20 border-purple-500/50 text-purple-300'
                      : 'bg-slate-800/50 border-slate-700/50 text-gray-300 hover:bg-slate-800/70'
                  }`}
                >
                  <h3 className="font-semibold mb-2">Auto Detect</h3>
                  <p className="text-sm opacity-80">
                    Automatically detects watermark type
                  </p>
                </button>
                <button
                  onClick={() => setVerificationMode('advanced')}
                  className={`p-4 rounded-lg border transition-all ${
                    verificationMode === 'advanced'
                      ? 'bg-cyan-600/20 border-cyan-500/50 text-cyan-300'
                      : 'bg-slate-800/50 border-slate-700/50 text-gray-300 hover:bg-slate-800/70'
                  }`}
                >
                  <h3 className="font-semibold mb-2">Advanced</h3>
                  <p className="text-sm opacity-80">
                    Check for advanced watermarks
                  </p>
                </button>
                <button
                  onClick={() => setVerificationMode('simple')}
                  className={`p-4 rounded-lg border transition-all ${
                    verificationMode === 'simple'
                      ? 'bg-blue-600/20 border-blue-500/50 text-blue-300'
                      : 'bg-slate-800/50 border-slate-700/50 text-gray-300 hover:bg-slate-800/70'
                  }`}
                >
                  <h3 className="font-semibold mb-2">Simple</h3>
                  <p className="text-sm opacity-80">
                    Check for basic watermarks
                  </p>
                </button>
              </div>
            </motion.div>

            {/* Image Upload Area */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              {!selectedImage ? (
                <div className="bg-background/80 backdrop-blur-lg border-2 border-dashed border-border/50 rounded-2xl p-12 text-center">
                  <div className="flex flex-col items-center gap-4">
                    <Search className="w-16 h-16 text-muted-foreground" />
                    <div>
                      <h3 className="text-xl font-semibold text-foreground mb-2">Select Image to Verify</h3>
                      <p className="text-muted-foreground mb-6">
                        Upload an image to analyze its authenticity and detect watermarks
                      </p>
                    </div>
                    <div className="flex gap-4">
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
                      >
                        <Upload className="w-4 h-4" />
                        Upload File
                      </button>
                      <button
                        onClick={handleCameraCapture}
                        className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center gap-2"
                      >
                        <Camera className="w-4 h-4" />
                        Use Camera
                      </button>

                      {/* Camera Capture Modal */}
                      {showCameraModal && (
                        <CameraCapture
                          onCapture={handleCameraCaptureComplete}
                          onClose={() => setShowCameraModal(false)}
                        />
                      )}
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                </div>
              ) : (
                <div className="space-y-6">
                  <div className="bg-background/80 backdrop-blur-lg border border-border/50 rounded-2xl p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-foreground">Selected Image</h3>
                      <button
                        onClick={resetForm}
                        className="text-slate-400 hover:text-white transition-colors"
                      >
                        Clear
                      </button>
                    </div>
                    <div className="bg-slate-900/50 rounded-lg overflow-hidden">
                      <img
                        src={selectedImage}
                        alt="Selected for verification"
                        className="w-full h-64 object-contain"
                      />
                    </div>
                    {selectedFileName && (
                      <p className="text-sm text-muted-foreground mt-2">{selectedFileName}</p>
                    )}
                  </div>

                  <div className="flex justify-center">
                    <button
                      onClick={processVerification}
                      disabled={isProcessing}
                      className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 text-white px-8 py-4 rounded-lg transition-all flex items-center gap-3 text-lg font-semibold"
                    >
                      <Search className="w-5 h-5" />
                      {isProcessing ? 'Analyzing...' : 'Verify Authenticity'}
                    </button>
                  </div>
                </div>
              )}
            </motion.div>
          </>
        ) : (
          /* Results Section - IMPROVED UI */
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {!verificationResult.error ? (
              <div className="bg-background/80 backdrop-blur-lg border border-border/50 rounded-2xl p-8">
                {/* Status Header */}
                <div className="text-center mb-8">
                  <div className={`inline-flex items-center px-6 py-3 rounded-full mb-4 ${
                    verificationResult.securityStatus === 'Authentic' 
                      ? 'bg-green-500/20 border-green-500/30' 
                      : verificationResult.securityStatus === 'External Source'
                        ? 'bg-blue-500/20 border-blue-500/30'
                        : verificationResult.securityStatus === 'Digital Capture'
                          ? 'bg-yellow-500/20 border-yellow-500/30'
                          : verificationResult.securityStatus === 'Modified'
                            ? 'bg-orange-500/20 border-orange-500/30'
                            : 'bg-red-500/20 border-red-500/30'
                  }`}>
                    {verificationResult.securityStatus === 'Authentic' ? (
                      <CheckCircle className="w-6 h-6 text-green-400 mr-2" />
                    ) : (
                      <AlertCircle className="w-6 h-6 text-red-400 mr-2" />
                    )}
                    <span className={`font-bold text-lg ${
                      verificationResult.securityStatus === 'Authentic' 
                        ? 'text-green-400' 
                        : verificationResult.securityStatus === 'External Source'
                          ? 'text-blue-400'
                          : verificationResult.securityStatus === 'Digital Capture'
                            ? 'text-yellow-400'
                            : verificationResult.securityStatus === 'Modified'
                              ? 'text-orange-400'
                              : 'text-red-400'
                    }`}>
                      {verificationResult.securityStatus}
                    </span>
                  </div>
                </div>

                {/* Uploaded image preview — above VERIFICATION REPORT */}
                {selectedImage && (
                  <div className="flex flex-col items-center gap-3 mb-6">
                    <p className="text-xs uppercase tracking-widest text-muted-foreground font-medium">
                      Uploaded Image Preview
                    </p>
                    <img
                      src={selectedImage}
                      alt="Uploaded preview"
                      className="max-h-[300px] w-auto rounded-xl object-contain shadow-2xl ring-1 ring-white/10"
                    />
                    {selectedFileName && (
                      <p className="text-xs text-muted-foreground">{selectedFileName}</p>
                    )}
                  </div>
                )}

                {/* IMPROVED VERIFICATION REPORT */}
                <div className="bg-accent/30 rounded-xl p-6">
                  <h3 className="text-xl font-bold text-foreground mb-6 text-center">VERIFICATION REPORT</h3>
                  
                  <div className="space-y-4">
                    {/* PINIT Encryption */}
                    <div className="flex justify-between items-center bg-background/50 rounded-lg p-4">
                      <span className="text-sm font-medium text-muted-foreground">PINIT Encryption:</span>
                      <span className={`text-lg font-bold ${
                        verificationResult.pinitEncrypted ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {verificationResult.pinitEncrypted ? 'YES' : 'NO'}
                      </span>
                    </div>
                    
                    {/* Camera Captured */}
                    <div className="flex justify-between items-center bg-background/50 rounded-lg p-4">
                      <span className="text-sm font-medium text-muted-foreground">Camera Captured:</span>
                      <span className={`text-lg font-bold ${
                        verificationResult.cameraCaptured ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {verificationResult.cameraCaptured ? 'YES' : 'NO'}
                      </span>
                    </div>
                    
                    {/* Image Source - MEANINGFUL CATEGORY */}
                    <div className="flex justify-between items-center bg-background/50 rounded-lg p-4">
                      <span className="text-sm font-medium text-muted-foreground">Image Source:</span>
                      <span className={`text-lg font-bold text-foreground capitalize`}>
                        {verificationResult.imageType}
                      </span>
                    </div>
                    
                    {/* Security Status */}
                    <div className="flex justify-between items-center bg-background/50 rounded-lg p-4">
                      <span className="text-sm font-medium text-muted-foreground">Security Status:</span>
                      <span className={`text-lg font-bold ${getSecurityStatusColor(verificationResult.securityStatus)}`}>
                        {verificationResult.securityStatus}
                      </span>
                    </div>
                    
                    {/* Confidence */}
                    <div className="flex justify-between items-center bg-background/50 rounded-lg p-4">
                      <span className="text-sm font-medium text-muted-foreground">Confidence:</span>
                      <span className={`text-lg font-bold ${getConfidenceColor(verificationResult.confidence)}`}>
                        {Math.round(verificationResult.confidence)}%
                      </span>
                    </div>
                  </div>

                  {/* ADDITIONAL ANALYSIS DETAILS */}
                  <div className="mt-6 space-y-4">
                    <h4 className="text-lg font-semibold text-foreground mb-4">Analysis Details</h4>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Detection Type */}
                      <div className="bg-background/50 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Shield className="w-5 h-5 text-blue-400" />
                          <span className="text-sm font-medium text-muted-foreground">Detection Type</span>
                        </div>
                        <div className={`text-lg font-bold ${getSecurityStatusColor(verificationResult.securityStatus)}`}>
                          {verificationResult.detectionType || 'Unknown'}
                        </div>
                      </div>
                      
                      {/* Risk Level */}
                      <div className="bg-background/50 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertTriangle className="w-5 h-5 text-orange-400" />
                          <span className="text-sm font-medium text-muted-foreground">Risk Level</span>
                        </div>
                        <div className={`text-lg font-bold ${getRiskLevelColor(verificationResult.riskLevel || 'Medium')}`}>
                          {verificationResult.riskLevel || 'Medium'}
                        </div>
                      </div>
                      
                      {/* Metadata Status */}
                      <div className="bg-background/50 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="w-5 h-5 text-purple-400" />
                          <span className="text-sm font-medium text-muted-foreground">Metadata Status</span>
                        </div>
                        <div className={`text-lg font-bold ${getSecurityStatusColor(verificationResult.securityStatus)}`}>
                          {verificationResult.metadataStatus}
                        </div>
                      </div>
                      
                      {/* Compression Status */}
                      <div className="bg-background/50 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <AlertCircle className="w-5 h-5 text-yellow-400" />
                          <span className="text-sm font-medium text-muted-foreground">Compression Status</span>
                        </div>
                        <div className={`text-lg font-bold ${getSecurityStatusColor(verificationResult.securityStatus)}`}>
                          {verificationResult.compressionDetected ? 'Detected' : 'Not Detected'}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* ═══════════════════════════════════════════════════════════
                    FORENSIC META ANALYSIS  v3
                ═══════════════════════════════════════════════════════════ */}
                {(() => {
                  const fr = verificationResult.analysis?.forensicReport;
                  const fs = fr?.forensicSignals;
                  const reliabilityScore = fr?.reliabilityScore ?? 0;

                  // Helper: render one signal confidence bar
                  const SignalBar = ({
                    label, score, confidence, reliability, isAI
                  }: { label: string; score: number; confidence: number; reliability: number; isAI: boolean }) => {
                    const pct = Math.round(score * 100);
                    const confPct = Math.round(confidence * 100);
                    const relPct  = Math.round(reliability * 100);
                    const barColor = isAI
                      ? score > 0.65 ? 'from-red-600 to-rose-400'
                        : score > 0.35 ? 'from-orange-600 to-amber-400'
                        : 'from-slate-600 to-slate-500'
                      : score > 0.60 ? 'from-emerald-600 to-green-400'
                        : score > 0.35 ? 'from-cyan-700 to-cyan-400'
                        : 'from-slate-600 to-slate-500';
                    const textColor = isAI
                      ? score > 0.65 ? 'text-rose-400' : score > 0.35 ? 'text-amber-400' : 'text-slate-500'
                      : score > 0.60 ? 'text-emerald-400' : score > 0.35 ? 'text-cyan-400' : 'text-slate-500';
                    return (
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-400 w-36 shrink-0 leading-tight">{label}</span>
                        <div className="flex-1 relative h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className={`absolute inset-y-0 left-0 rounded-full bg-gradient-to-r ${barColor} transition-all duration-700`}
                            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
                          />
                        </div>
                        <span className={`text-[10px] font-mono font-bold w-8 text-right ${textColor}`}>{pct}%</span>
                        <span className="text-[9px] text-slate-600 w-14 text-right shrink-0">
                          c:{confPct}% r:{relPct}%
                        </span>
                      </div>
                    );
                  };

                  return (
                    <div className="mt-6 bg-slate-900/70 border border-cyan-500/25 rounded-xl p-6 space-y-6">

                      {/* ── Header ── */}
                      <div className="flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-cyan-400" />
                        <span className="text-xs font-bold text-cyan-400 tracking-[0.2em] uppercase">
                          Forensic Meta Analysis · v3 Probabilistic Fusion
                        </span>
                        <div className="flex-1 h-px bg-cyan-500/20 ml-2" />
                        <div className="flex items-center gap-1.5">
                          <span className="text-[9px] text-slate-500 uppercase tracking-wider">Signal Quality</span>
                          <span className={`text-xs font-bold ${
                            reliabilityScore >= 70 ? 'text-emerald-400' :
                            reliabilityScore >= 45 ? 'text-amber-400' : 'text-slate-400'
                          }`}>{reliabilityScore}%</span>
                        </div>
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse ml-1" />
                      </div>

                      {/* ── Fusion Scores ── */}
                      <div>
                        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em] mb-3">
                          Fusion Probability Scores
                        </p>
                        <div className="space-y-2">
                          {[
                            {
                              label: 'AI Generated',
                              value: verificationResult.aiProbability,
                              barClass: verificationResult.aiProbability >= 60 ? 'from-red-600 to-rose-400'
                                       : verificationResult.aiProbability >= 35 ? 'from-orange-600 to-amber-400'
                                       : 'from-slate-600 to-slate-500',
                              textClass: verificationResult.aiProbability >= 60 ? 'text-rose-400'
                                        : verificationResult.aiProbability >= 35 ? 'text-amber-400' : 'text-slate-400',
                              note: 'Threshold ≥ 60% + ≥2 strong signals for AI classification',
                            },
                            {
                              label: 'Camera Captured',
                              value: verificationResult.cameraProbability,
                              barClass: verificationResult.cameraProbability >= 43 ? 'from-emerald-600 to-green-400' : 'from-slate-600 to-slate-500',
                              textClass: verificationResult.cameraProbability >= 43 ? 'text-emerald-400' : 'text-slate-400',
                              note: 'Threshold ≥ 43% — robust to WhatsApp/EXIF stripping',
                            },
                            {
                              label: 'Screenshot (diag.)',
                              value: verificationResult.screenshotProbability,
                              barClass: 'from-amber-700 to-yellow-500',
                              textClass: 'text-amber-400',
                              note: 'Diagnostic only — does not affect classification',
                            },
                            {
                              label: 'Edited (diag.)',
                              value: verificationResult.editedProbability,
                              barClass: 'from-orange-700 to-amber-500',
                              textClass: 'text-orange-400',
                              note: 'Diagnostic only — does not affect classification',
                            },
                          ].map(({ label, value, barClass, textClass, note }) => (
                            <div key={label}>
                              <div className="flex items-center gap-3">
                                <span className="text-xs text-slate-400 w-32 shrink-0">{label}</span>
                                <div className="flex-1 relative h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                  <div
                                    className={`absolute inset-y-0 left-0 rounded-full bg-gradient-to-r ${barClass} transition-all duration-700`}
                                    style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
                                  />
                                </div>
                                <span className={`text-xs font-mono font-bold w-10 text-right ${textClass}`}>{Math.round(value)}%</span>
                              </div>
                              <p className="text-[9px] text-slate-600 ml-[140px] mt-0.5">{note}</p>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* ── AI Signal Confidence Bars ── */}
                      {fs && (
                        <div>
                          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em] mb-1">
                            AI Detection Signals
                            <span className="ml-2 text-rose-400/60 normal-case">high score = AI indicator</span>
                          </p>
                          <p className="text-[9px] text-slate-600 mb-3">c = measurement confidence · r = signal reliability</p>
                          <div className="space-y-1.5">
                            <SignalBar label="Noise Floor Deficit" score={fs.ai.noiseFloorDeficit.score}   confidence={fs.ai.noiseFloorDeficit.confidence}   reliability={fs.ai.noiseFloorDeficit.reliability}   isAI={true} />
                            <SignalBar label="Microtexture Entropy" score={fs.ai.microtextureEntropy.score} confidence={fs.ai.microtextureEntropy.confidence} reliability={fs.ai.microtextureEntropy.reliability} isAI={true} />
                            <SignalBar label="Edge Uniformity"     score={fs.ai.edgeUniformity.score}      confidence={fs.ai.edgeUniformity.confidence}      reliability={fs.ai.edgeUniformity.reliability}      isAI={true} />
                            <SignalBar label="Pattern Repetition"  score={fs.ai.patternRepetition.score}   confidence={fs.ai.patternRepetition.confidence}   reliability={fs.ai.patternRepetition.reliability}   isAI={true} />
                            <SignalBar label="Symmetry Bias"       score={fs.ai.symmetryBias.score}        confidence={fs.ai.symmetryBias.confidence}        reliability={fs.ai.symmetryBias.reliability}        isAI={true} />
                            <SignalBar label="Frequency Deficit"   score={fs.ai.frequencyDeficit.score}    confidence={fs.ai.frequencyDeficit.confidence}    reliability={fs.ai.frequencyDeficit.reliability}    isAI={true} />
                          </div>
                        </div>
                      )}

                      {/* ── Camera Signal Confidence Bars ── */}
                      {fs && (
                        <div>
                          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em] mb-1">
                            Camera Detection Signals
                            <span className="ml-2 text-emerald-400/60 normal-case">high score = camera indicator</span>
                          </p>
                          <p className="text-[9px] text-slate-600 mb-3">c = measurement confidence · r = signal reliability</p>
                          <div className="space-y-1.5">
                            <SignalBar label="Sensor Noise Residual" score={fs.camera.sensorNoiseResidual.score} confidence={fs.camera.sensorNoiseResidual.confidence} reliability={fs.camera.sensorNoiseResidual.reliability} isAI={false} />
                            <SignalBar label="JPEG Naturalness"      score={fs.camera.jpegNaturalness.score}     confidence={fs.camera.jpegNaturalness.confidence}     reliability={fs.camera.jpegNaturalness.reliability}     isAI={false} />
                            <SignalBar label="CFA Demosaic"          score={fs.camera.cfaDemosaic.score}         confidence={fs.camera.cfaDemosaic.confidence}         reliability={fs.camera.cfaDemosaic.reliability}         isAI={false} />
                            <SignalBar label="Chromatic Aberration"  score={fs.camera.chromaticAberration.score} confidence={fs.camera.chromaticAberration.confidence} reliability={fs.camera.chromaticAberration.reliability} isAI={false} />
                            <SignalBar label="Edge Randomness"       score={fs.camera.edgeRandomness.score}      confidence={fs.camera.edgeRandomness.confidence}      reliability={fs.camera.edgeRandomness.reliability}      isAI={false} />
                            <SignalBar label="Face Region Noise"     score={fs.camera.faceRegionNoise.score}     confidence={fs.camera.faceRegionNoise.confidence}     reliability={fs.camera.faceRegionNoise.reliability}     isAI={false} />
                          </div>
                        </div>
                      )}

                      {/* ── Decision Explanation ── */}
                      {fr?.fusionDebug && (
                        <div className="bg-slate-800/40 border border-slate-700/40 rounded-lg p-4">
                          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em] mb-2">
                            Decision Explanation
                          </p>
                          <p className="text-[11px] text-slate-300 leading-relaxed mb-2">
                            {(fr.fusionDebug as any).finalDecisionReason}
                          </p>
                          {(fr.fusionDebug as any).falsePositiveProtectionApplied && (
                            <div className="flex items-start gap-2 mt-2 p-2 bg-amber-500/10 border border-amber-500/20 rounded">
                              <AlertTriangle className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
                              <p className="text-[10px] text-amber-300 leading-relaxed">
                                <strong>False-positive protection active:</strong> {(fr.fusionDebug as any).suppressionReason}.
                                Real camera photos require fewer AI signals to be overridden.
                              </p>
                            </div>
                          )}
                          {(fr.fusionDebug as any).dominantSignals?.length > 0 && (
                            <div className="mt-2">
                              <p className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Dominant signals</p>
                              <div className="flex flex-wrap gap-1">
                                {(fr.fusionDebug as any).dominantSignals.map((s: string) => (
                                  <span key={s} className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                                    s.startsWith('AI:') ? 'bg-rose-900/40 text-rose-300 border border-rose-800/40' : 'bg-emerald-900/40 text-emerald-300 border border-emerald-800/40'
                                  }`}>{s}</span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* ── Forensic Warnings ── */}
                      {(() => {
                        const warnings: string[] = [];
                        if (verificationResult.compressionDetected)
                          warnings.push('Heavy re-compression detected — JPEG-based signals may be less reliable.');
                        if (!verificationResult.analysis?.metadata.hasExif)
                          warnings.push('No EXIF metadata — common for WhatsApp, Telegram, and screenshot saves. This alone does NOT indicate AI.');
                        if (reliabilityScore < 45)
                          warnings.push('Low overall signal quality — result may be inconclusive. Try a higher-resolution image.');
                        if (verificationResult.imageType === 'unknown')
                          warnings.push('Classification inconclusive — signals do not agree. Manual review recommended.');
                        if (fs && fs.ai.frequencyDeficit.score > 0.80 && verificationResult.compressionDetected)
                          warnings.push('Frequency deficit signal is high but image is also heavily compressed — this combination is common in WhatsApp camera images and does NOT confirm AI generation.');
                        return warnings.length > 0 ? (
                          <div className="space-y-1.5">
                            <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em]">Forensic Warnings</p>
                            {warnings.map((w, i) => (
                              <div key={i} className="flex items-start gap-2 p-2 bg-amber-500/8 border border-amber-600/15 rounded">
                                <AlertTriangle className="w-3 h-3 text-amber-500 mt-0.5 shrink-0" />
                                <p className="text-[10px] text-amber-200/70 leading-relaxed">{w}</p>
                              </div>
                            ))}
                          </div>
                        ) : null;
                      })()}

                      {/* ── System Status Row ── */}
                      <div>
                        <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-[0.15em] mb-2">System Status</p>
                        <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                          <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Metadata</p>
                            <div className="flex items-center gap-1.5">
                              <div className={`w-2 h-2 rounded-full ${verificationResult.metadataStatus === 'original' ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                              <span className={`text-xs font-semibold ${verificationResult.metadataStatus === 'original' ? 'text-emerald-400' : 'text-amber-400'}`}>
                                {verificationResult.metadataStatus === 'original' ? 'EXIF Intact' : 'No EXIF'}
                              </span>
                            </div>
                            <p className="text-[10px] text-slate-600 mt-1">
                              {verificationResult.metadataStatus === 'original' ? 'Camera metadata present' : 'Stripped (common for shared images)'}
                            </p>
                          </div>
                          <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Compression</p>
                            <div className="flex items-center gap-1.5">
                              <div className={`w-2 h-2 rounded-full ${verificationResult.compressionDetected ? 'bg-amber-400' : 'bg-emerald-400'}`} />
                              <span className={`text-xs font-semibold ${verificationResult.compressionDetected ? 'text-amber-400' : 'text-emerald-400'}`}>
                                {verificationResult.compressionDetected ? 'Recompressed' : 'Clean'}
                              </span>
                            </div>
                            <p className="text-[10px] text-slate-600 mt-1">
                              {verificationResult.compressionDetected ? 'WhatsApp/social re-encoding' : 'No re-encoding detected'}
                            </p>
                          </div>
                          <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg p-3">
                            <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">FP Protection</p>
                            <div className="flex items-center gap-1.5">
                              <div className={`w-2 h-2 rounded-full ${verificationResult.suppressionTriggered ? 'bg-amber-400' : 'bg-emerald-400'}`} />
                              <span className={`text-xs font-semibold ${verificationResult.suppressionTriggered ? 'text-amber-400' : 'text-emerald-400'}`}>
                                {verificationResult.suppressionTriggered ? 'Active' : 'Not Needed'}
                              </span>
                            </div>
                            <p className="text-[10px] text-slate-600 mt-1">
                              {verificationResult.suppressionTriggered
                                ? 'AI score dampened — insufficient independent signals'
                                : 'AI signals met independent-signal requirement'}
                            </p>
                          </div>
                        </div>
                      </div>


                    {/* ── Deep Learning Inference Panel ── */}
                    {(() => {
                      const dl = (fr as any)?.dlResult;
                      const cal = (fr as any)?.dlCalibration;
                      const weights = (fr as any)?.dlFusionWeights as Record<string, number> | undefined;
                      const domSigs = (fr as any)?.dlDominantSignals as string[] | undefined;
                      const fusedPct = (fr as any)?.fusedAiProbability as number | undefined;
                      const dlMs = (fr as any)?.dlProcessingMs as number | undefined;
                      const dlAvail = (fr as any)?.dlAvailable as boolean | undefined;
                      if (!dl) return null;
                      const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
                      const barColor = (v: number) =>
                        v > 0.65 ? 'bg-rose-500' : v < 0.40 ? 'bg-emerald-500' : 'bg-amber-500';
                      const branches: [string, number, string][] = [
                        ['CNN (RGB image)',     dl.cnn_score,      'DL branch — spatial texture & colour (35%)'],
                        ['Residual (noise)',    dl.residual_score, 'DL branch — sensor noise fingerprint (20%)'],
                        ['FFT (frequency)',     dl.fft_score,      'DL branch — frequency-domain decay (15%)'],
                        ['Frequency heuristic',dl.forensic_score, 'Classical spectral analysis (20%)'],
                        ['Metadata',           dl.metadata_score, 'EXIF / file-level signal (10%)'],
                      ];
                      return (
                        <div className="bg-slate-800/40 border border-violet-700/30 rounded-lg p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <p className="text-[10px] font-semibold text-violet-400 uppercase tracking-[0.15em]">
                              Deep Learning Inference
                            </p>
                            <div className="flex items-center gap-2">
                              <div className={`w-2 h-2 rounded-full ${dlAvail !== false ? 'bg-violet-400' : 'bg-slate-600'}`} />
                              <span className="text-[10px] text-slate-500">
                                {cal?.model_backend ?? 'heuristic'}{cal?.has_trained_weights ? ' · trained' : ' · proxy'}
                                {dlMs != null ? ` · ${dlMs.toFixed(0)}ms` : ''}
                              </span>
                            </div>
                          </div>

                          {/* Fused probability */}
                          {fusedPct != null && (
                            <div className="flex items-center gap-3">
                              <span className="text-[10px] text-slate-500 w-28 shrink-0">Fused AI score</span>
                              <div className="flex-1 h-2 bg-slate-700 rounded-full overflow-hidden">
                                <div className={`h-full rounded-full transition-all ${barColor(fusedPct / 100)}`}
                                  style={{ width: `${fusedPct}%` }} />
                              </div>
                              <span className={`text-[11px] font-mono font-semibold w-10 text-right ${
                                fusedPct > 55 ? 'text-rose-400' : fusedPct < 45 ? 'text-emerald-400' : 'text-amber-400'
                              }`}>{fusedPct.toFixed(1)}%</span>
                            </div>
                          )}

                          {/* Branch bars */}
                          <div className="space-y-1.5 pt-1">
                            {branches.map(([label, score, tooltip]) => (
                              <div key={label} className="flex items-center gap-3" title={tooltip}>
                                <span className="text-[9px] text-slate-500 w-28 shrink-0 truncate">{label}</span>
                                <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                                  <div className={`h-full rounded-full transition-all ${barColor(score)}`}
                                    style={{ width: `${score * 100}%` }} />
                                </div>
                                <span className="text-[10px] font-mono text-slate-400 w-10 text-right">{pct(score)}</span>
                              </div>
                            ))}
                          </div>

                          {/* Dominant DL signals */}
                          {domSigs && domSigs.length > 0 && (
                            <div>
                              <p className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">DL signals</p>
                              <div className="space-y-0.5">
                                {domSigs.slice(0, 4).map((s, i) => (
                                  <p key={i} className="text-[9px] text-slate-400 leading-relaxed">• {s}</p>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Fusion weights */}
                          {weights && Object.keys(weights).length > 0 && (
                            <div>
                              <p className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Fusion weights</p>
                              <div className="flex flex-wrap gap-1">
                                {Object.entries(weights).map(([k, v]) => (
                                  <span key={k} className="text-[9px] font-mono px-1.5 py-0.5 bg-slate-700/60 border border-slate-600/40 rounded text-slate-400">
                                    {k}={`${(v * 100).toFixed(0)}%`}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })()}

                    </div>
                  );
                })()}
                {/* ─── end forensic meta analysis ─── */}

                {/* ── Report Wrong Detection ──────────────────────────────── */}
                <div className="mt-6">
                  {reportState === 'idle' && (
                    <div className="flex justify-center">
                      <button
                        onClick={() => setReportState('selecting')}
                        className="text-xs text-slate-500 hover:text-amber-400 underline underline-offset-2 transition-colors duration-150"
                      >
                        ⚑ Report Wrong Detection
                      </button>
                    </div>
                  )}

                  {reportState === 'selecting' && (
                    <div className="bg-slate-900/60 border border-amber-600/25 rounded-xl p-5 space-y-3">
                      <p className="text-xs font-semibold text-amber-400 uppercase tracking-widest text-center">
                        What is the correct label for this image?
                      </p>
                      <div className="flex flex-wrap gap-3 justify-center">
                        <button
                          onClick={() => reportWrongDetection('camera')}
                          className="px-4 py-2 rounded-lg bg-emerald-800/30 border border-emerald-600/40 text-emerald-300 text-xs font-semibold hover:bg-emerald-700/50 transition-colors"
                        >
                          📷 Real Camera Photo
                        </button>
                        <button
                          onClick={() => reportWrongDetection('ai')}
                          className="px-4 py-2 rounded-lg bg-rose-800/30 border border-rose-600/40 text-rose-300 text-xs font-semibold hover:bg-rose-700/50 transition-colors"
                        >
                          🤖 AI Generated
                        </button>
                        <button
                          onClick={() => setReportState('idle')}
                          className="px-4 py-2 rounded-lg bg-slate-700/30 border border-slate-600/40 text-slate-400 text-xs font-semibold hover:bg-slate-600/50 transition-colors"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}

                  {reportState === 'submitting' && (
                    <div className="flex items-center justify-center gap-2 text-xs text-slate-400 py-2">
                      <div className="w-3 h-3 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                      Saving report…
                    </div>
                  )}

                  {(reportState === 'success' || reportState === 'error') && (
                    <div className={`flex items-center justify-center gap-2 text-xs font-medium py-2 ${
                      reportState === 'success' ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      <span>{reportState === 'success' ? '✓' : '✗'}</span>
                      <span>{reportMessage}</span>
                    </div>
                  )}
                </div>
                {/* ── end report wrong detection ── */}

              </div>
            ) : (
              <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-6 text-center">
                <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-red-400 mb-2">Verification Failed</h3>
                <p className="text-red-300 mb-4">{verificationResult.error}</p>
                <Button
                  onClick={resetForm}
                  variant="outline"
                  className="border-red-500/50 text-red-400 hover:bg-red-500/10"
                >
                  Try Again
                </Button>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default VerifyProof;
