import fs from 'fs';

let html = fs.readFileSync('trending/index.html', 'utf8');

// Inject tailwind CDN to ensure our new classes work
if (!html.includes('cdn.tailwindcss.com')) {
    html = html.replace('</head>', '<script src="https://cdn.tailwindcss.com"></script></head>');
}

const mockFeed = `
<div class="h-full border-t border-gray-200 dark:border-gray-800 mt-4">
    <!-- Create Post Area -->
    <div class="p-4 border-b border-gray-200 dark:border-gray-800">
        <div class="flex gap-4">
            <div class="w-12 h-12 rounded-full bg-gray-300 dark:bg-gray-700 flex-shrink-0"></div>
            <div class="flex-1">
                <textarea class="w-full bg-transparent text-lg outline-none resize-none placeholder-gray-500 min-h-[80px]" placeholder="Nələr baş verir?"></textarea>
                <div class="flex justify-between items-center pt-2 border-t border-gray-200 dark:border-gray-800 mt-2">
                    <div class="flex gap-2">
                        <!-- S2T Button -->
                        <button class="flex items-center gap-2 text-blue-500 hover:bg-blue-500/10 px-3 py-1.5 rounded-full font-medium transition" onclick="alert('S2T Started!')">
                            📷 <span class="text-sm">Sign to Text (S2T)</span>
                        </button>
                    </div>
                    <button class="bg-blue-500 text-white font-bold py-1.5 px-5 rounded-full hover:bg-blue-600 transition">Paylaş</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Mock Post -->
    <div class="p-4 border-b border-gray-200 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900 transition">
        <div class="flex gap-4">
            <div class="w-12 h-12 rounded-full bg-blue-500 flex-shrink-0"></div>
            <div class="flex-1">
                <div class="flex items-center justify-between">
                    <div class="font-bold flex items-center gap-1">Test User <span class="text-gray-500 font-normal text-sm">@testuser · 1s</span></div>
                </div>
                <p class="mt-2 text-[15px]">Bu NSosyal klonunda eksperimental modullar test edilir. Əlçatanlıq çox vacibdir!</p>
                
                <!-- Post Actions -->
                <div class="flex justify-between mt-3 text-gray-500 max-w-md">
                    <button class="hover:text-blue-500 transition">💬 5</button>
                    <button class="hover:text-green-500 transition">🔁 2</button>
                    <button class="hover:text-red-500 transition">❤️ 12</button>
                    <!-- T2S Button -->
                    <button class="flex items-center gap-1 text-purple-500 hover:bg-purple-500/10 px-2 py-1 rounded-full font-medium transition" onclick="alert('T2S Started! Avatar will spell this text.')">
                        🧑‍🦽 <span class="text-sm">Text to Sign (T2S)</span>
                    </button>
                </div>

                <!-- Comment Area -->
                <div class="mt-4 flex gap-3">
                    <div class="w-8 h-8 rounded-full bg-gray-300 dark:bg-gray-700 flex-shrink-0"></div>
                    <div class="flex-1 flex gap-2 items-center bg-gray-100 dark:bg-gray-800 border border-transparent focus-within:border-blue-500 rounded-full px-4 py-1">
                        <input type="text" class="bg-transparent flex-1 outline-none text-sm placeholder-gray-500" placeholder="Rəyini yaz...">
                        <!-- S2T Button for Comment -->
                        <button class="text-blue-500 hover:bg-blue-500/10 font-medium text-sm px-3 py-1 rounded-full flex items-center gap-1" onclick="alert('S2T Started for comment!')">
                            📷 S2T
                        </button>
                        <button class="text-white font-bold text-sm bg-blue-500 rounded-full px-4 py-1.5 hover:bg-blue-600">Göndər</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
`;

// Replace the spinner container with the mock feed
html = html.replace(/<div class="h-full">.*?<\/div><\/main>/, mockFeed + '</main>');

fs.writeFileSync('trending/index.html', html);
