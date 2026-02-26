import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
    appId: 'com.nexus.ai.mobile',
    appName: 'NEXUS AI',
    webDir: 'www',
    server: {
        // Allow loading external URLs (the NEXUS server)
        allowNavigation: ['*'],
        cleartext: true, // Allow HTTP (non-HTTPS) connections for local servers
    },
    android: {
        allowMixedContent: true,
        backgroundColor: '#060810',
        buildOptions: {
            releaseType: 'APK',
        },
    },
    plugins: {
        SplashScreen: {
            launchShowDuration: 2000,
            backgroundColor: '#060810',
            showSpinner: true,
            spinnerColor: '#00d4ff',
            androidSplashResourceName: 'splash',
        },
        StatusBar: {
            style: 'DARK',
            backgroundColor: '#060810',
        },
    },
};

export default config;
