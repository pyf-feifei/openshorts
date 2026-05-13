import React, { useState, useEffect } from 'react';
import { Key, Eye, EyeOff, Check, Globe, Cookie, Loader2 } from 'lucide-react';
import { useI18n } from '../i18n/I18nProvider';
import { getApiUrl } from '../config';

export default function KeyInput({ onKeySet, savedKey, onBaseUrlSet, savedBaseUrl, children }) {
    const { t } = useI18n();
    const [key, setKey] = useState(savedKey || '');
    const [baseUrl, setBaseUrl] = useState(savedBaseUrl || '');
    const [isVisible, setIsVisible] = useState(false);
    const [isSaved, setIsSaved] = useState(!!savedKey);
    const [cookies, setCookies] = useState('');
    const [cookiesStatus, setCookiesStatus] = useState(null);
    const [cookiesMessage, setCookiesMessage] = useState('');
    const [isSavingCookies, setIsSavingCookies] = useState(false);
    const [isVerifyingCookies, setIsVerifyingCookies] = useState(false);

    useEffect(() => {
        if (savedKey) setKey(savedKey);
    }, [savedKey]);

    useEffect(() => {
        setBaseUrl(savedBaseUrl || '');
    }, [savedBaseUrl]);

    useEffect(() => {
        fetch(getApiUrl('/api/settings/youtube-cookies'))
            .then((res) => res.ok ? res.json() : null)
            .then((data) => setCookiesStatus(data))
            .catch(() => setCookiesStatus(null));
    }, []);

    const saveCookiesValue = async (value) => {
        const normalized = value.trim();
        if (!normalized) {
            setCookiesMessage('请先粘贴或上传 YouTube cookies 内容');
            return;
        }
        setIsSavingCookies(true);
        setCookiesMessage('');
        try {
            const res = await fetch(getApiUrl('/api/settings/youtube-cookies'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cookies: normalized }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                setCookiesStatus(data.configured !== undefined ? data : cookiesStatus);
                throw new Error(data.detail || '保存 cookies 失败');
            }
            setCookies('');
            setCookiesStatus(data);
            const strongFields = data.found_strong_login_cookies?.length ? `，登录字段：${data.found_strong_login_cookies.join(' / ')}` : '';
            setCookiesMessage(`已保存 YouTube cookies，大小 ${Math.round((data.size || 0) / 1024)} KB，识别 ${data.rows || 0} 条${strongFields}`);
        } catch (error) {
            setCookiesMessage(error.message);
        } finally {
            setIsSavingCookies(false);
        }
    };

    const handleSaveCookies = async () => {
        const value = cookies.trim();
        if (!value) {
            setCookiesMessage('请先粘贴 YouTube cookies 内容');
            return;
        }
        await saveCookiesValue(value);
    };

    const handleUploadCookies = async (event) => {
        const file = event.target.files?.[0];
        event.target.value = '';
        if (!file) return;
        try {
            const text = await file.text();
            await saveCookiesValue(text);
        } catch (error) {
            setCookiesMessage(error.message || '读取 cookies 文件失败');
        }
    };

    const handleVerifyCookies = async () => {
        setIsVerifyingCookies(true);
        setCookiesMessage('');
        try {
            const res = await fetch(getApiUrl('/api/settings/youtube-cookies/verify'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                const detail = data.detail || {};
                const message = typeof detail === 'string' ? detail : detail.message;
                throw new Error(message || 'YouTube cookies 验证失败');
            }
            setCookiesMessage(`验证通过：yt-dlp 可以解析 YouTube 视频${data.title ? `《${data.title}》` : ''}，可用格式 ${data.format_count || 0} 个。`);
        } catch (error) {
            setCookiesMessage(error.message || 'YouTube cookies 验证失败，请重新导出 cookies。');
        } finally {
            setIsVerifyingCookies(false);
        }
    };

    const handleSave = () => {
        if (key.trim().length > 0) {
            onKeySet(key.trim());
            onBaseUrlSet?.(baseUrl.trim().replace(/\/+$/, ''));
            setIsSaved(true);
        }
    };

    return (
        <>
        <div className="bg-surface border border-white/5 rounded-2xl p-6 mb-8 animate-[fadeIn_0.5s_ease-out]">
            <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-accent/20 rounded-lg text-accent">
                    <Key size={20} />
                </div>
                <h2 className="text-lg font-semibold">{t('keyInput.title')}</h2>
            </div>

            <div className="space-y-3">
                <div className="flex gap-3">
                    <div className="relative flex-1">
                        <input
                            type={isVisible ? "text" : "password"}
                            value={key}
                            onChange={(e) => {
                                setKey(e.target.value);
                                setIsSaved(false);
                            }}
                            placeholder="AIzaSy... 或 sk-..."
                            className="input-field pr-12 font-mono"
                        />
                        <button
                            onClick={() => setIsVisible(!isVisible)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-white transition-colors"
                        >
                            {isVisible ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                    </div>
                    <button
                        onClick={handleSave}
                        disabled={!key || isSaved}
                        className={`px-6 rounded-xl font-medium transition-all flex items-center gap-2 ${isSaved
                            ? 'bg-green-500/20 text-green-400 cursor-default'
                            : 'bg-primary hover:bg-blue-600 text-white shadow-lg shadow-primary/20'
                            }`}
                    >
                        {isSaved ? <><Check size={18} /> {t('common.ready')}</> : t('keyInput.setKey')}
                    </button>
                </div>

                <div className="relative">
                    <Globe size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
                    <input
                        type="url"
                        value={baseUrl}
                        onChange={(e) => {
                            setBaseUrl(e.target.value);
                            setIsSaved(false);
                        }}
                        placeholder="可选 Base URL，留空使用官方 Gemini，例如 https://gemini-balance-lite.hbd74900.workers.dev"
                        className="input-field pl-10 font-mono text-sm"
                    />
                </div>
            </div>
            {children}
            <p className="mt-3 text-xs text-zinc-500">
                {t('keyInput.localStorageNote')}
                <br />
                <a
                    href="https://aistudio.google.com/app/apikey"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline mt-1 inline-block"
                >
                    {t('keyInput.getKey')}
                </a>
            </p>
        </div>

        <div className="bg-surface border border-white/5 rounded-2xl p-6 mb-8 animate-[fadeIn_0.5s_ease-out]">
            <div className="flex items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-orange-500/20 rounded-lg text-orange-400">
                        <Cookie size={20} />
                    </div>
                    <div>
                        <h2 className="text-lg font-semibold">YouTube Cookies</h2>
                        <p className="text-xs text-zinc-500">用于绕过 YouTube bot 校验；如果页面提示 cookies 失效，请重新导出后粘贴保存。</p>
                    </div>
                </div>
                {cookiesStatus?.configured && (
                    <span className={`text-xs px-3 py-1 rounded-full border ${cookiesStatus.missing_warning ? 'bg-yellow-500/10 text-yellow-300 border-yellow-500/20' : 'bg-green-500/10 text-green-400 border-green-500/20'}`}>
                        已配置 · {Math.round((cookiesStatus.size || 0) / 1024)} KB
                    </span>
                )}
            </div>
            <textarea
                value={cookies}
                onChange={(e) => setCookies(e.target.value)}
                placeholder="粘贴 Netscape cookies.txt 内容，例如包含 .youtube.com 的多行 cookies"
                className="input-field min-h-[140px] font-mono text-xs resize-y"
            />
            <div className="flex flex-wrap items-center justify-between gap-3 mt-3">
                <p className="text-xs text-zinc-500">
                    建议从当前登录 YouTube 的浏览器重新导出 cookies.txt，旧 cookies 会因为浏览器轮换而失效。
                </p>
                <div className="flex items-center gap-2">
                    <label className="px-5 py-2 rounded-xl border border-white/10 hover:border-white/20 text-zinc-200 font-medium transition-colors whitespace-nowrap cursor-pointer">
                        上传 cookies.txt
                        <input
                            type="file"
                            accept=".txt,text/plain"
                            onChange={handleUploadCookies}
                            disabled={isSavingCookies}
                            className="hidden"
                        />
                    </label>
                    <button
                        onClick={handleSaveCookies}
                        disabled={isSavingCookies || !cookies.trim()}
                        className="px-5 py-2 rounded-xl bg-primary hover:bg-blue-600 disabled:bg-zinc-700 disabled:text-zinc-400 text-white font-medium transition-colors whitespace-nowrap"
                    >
                        {isSavingCookies ? '保存中...' : '保存粘贴内容'}
                    </button>
                    <button
                        onClick={handleVerifyCookies}
                        disabled={isVerifyingCookies || !cookiesStatus?.configured}
                        className="px-5 py-2 rounded-xl border border-green-500/30 bg-green-500/10 hover:bg-green-500/20 disabled:bg-zinc-800 disabled:text-zinc-500 text-green-300 font-medium transition-colors whitespace-nowrap inline-flex items-center gap-2"
                    >
                        {isVerifyingCookies && <Loader2 size={16} className="animate-spin" />}
                        {isVerifyingCookies ? '验证中...' : '验证 cookies'}
                    </button>
                </div>
            </div>
            {cookiesStatus?.missing_warning && (
                <p className="mt-3 text-xs text-yellow-300">
                    当前 cookies 可能不是完整登录 cookies：缺少 SID / SAPISID / LOGIN_INFO 等关键字段，YouTube 仍可能要求登录或 bot 校验。
                </p>
            )}
            {cookiesStatus?.rows !== undefined && (
                <p className="mt-2 text-xs text-zinc-500">
                    已识别 {cookiesStatus.rows} 条 cookies。
                    {cookiesStatus.found_strong_login_cookies?.length ? ` 已包含登录字段：${cookiesStatus.found_strong_login_cookies.join(' / ')}。` : ''}
                    {cookiesStatus.missing_strong_login_cookies?.length ? ` 缺少：${cookiesStatus.missing_strong_login_cookies.join(' / ')}。` : ''}
                    建议使用浏览器导出的完整 YouTube/Google 登录 cookies，或使用 cookies-from-browser 方式。
                </p>
            )}
            {cookiesMessage && <p className="mt-3 text-xs text-zinc-400">{cookiesMessage}</p>}
        </div>
        </>
    );
}
