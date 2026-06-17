import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { Fingerprint, FileSearch, AlertCircle, Search } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { listCases } from "@/lib/api";
import type { CaseResponse } from "@/lib/api";

const STATUS_COLORS: Record<string, string> = {
  open: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800",
  closed: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
  archived: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
};

function CaseCard({ caseItem }: { caseItem: CaseResponse }) {
  const statusClass = STATUS_COLORS[caseItem.status] ?? STATUS_COLORS.open;

  return (
    <Link
      to={`/cases/${caseItem.id}/compare`}
      className="block transition-colors hover:bg-muted/50 rounded-lg"
    >
      <Card className="border-border/60 cursor-pointer">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Fingerprint className="w-5 h-5 text-primary" />
              </div>
              <div>
                <CardTitle className="text-base font-semibold">
                  {caseItem.title}
                </CardTitle>
                <p className="text-xs text-muted-foreground font-mono mt-0.5">
                  {caseItem.case_number}
                </p>
              </div>
            </div>
            <Badge
              variant="outline"
              className={`text-xs font-medium capitalize ${statusClass}`}
            >
              {caseItem.status}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          {caseItem.description && (
            <p className="text-sm text-muted-foreground line-clamp-2 mb-2">
              {caseItem.description}
            </p>
          )}
          <p className="text-xs text-muted-foreground">
            Creado: {new Date(caseItem.created_at).toLocaleDateString("es-NI", {
              year: "numeric",
              month: "long",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
            })}
          </p>
        </CardContent>
      </Card>
    </Link>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="p-4 bg-muted/30 rounded-full mb-4">
        <FileSearch className="w-10 h-10 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-semibold text-foreground mb-1">
        No hay casos activos
      </h3>
      <p className="text-sm text-muted-foreground max-w-sm mb-6">
        Cree un nuevo caso forense o suba evidencia a un caso existente para
        comenzar el proceso de identificación.
      </p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 3 }).map((_, i) => (
        <Card key={i} className="border-border/60 animate-pulse">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-muted rounded-lg" />
              <div className="space-y-2 flex-1">
                <div className="h-4 bg-muted rounded w-2/3" />
                <div className="h-3 bg-muted rounded w-1/3" />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-3 bg-muted rounded w-1/4" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const navigate = useNavigate();

  const {
    data: caseList,
    isLoading,
    isError,
    error,
  } = useQuery({
    queryKey: ["cases"],
    queryFn: () => listCases("open"),
  });

  return (
    <div className="min-h-screen bg-background text-foreground p-8 font-sans dark">
      <div className="max-w-5xl mx-auto space-y-8">
        {/* Header */}
        <header className="flex items-center justify-between border-b border-border pb-6">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-primary/10 rounded-full">
              <Fingerprint className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Panel de Casos Forenses
              </h1>
              <p className="text-muted-foreground text-sm">
                Gestión de identificación dactilar — Sistema BioSecure Gov
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button
              variant="outline"
              size="lg"
              onClick={() => navigate("/search")}
              className="gap-2"
            >
              <Search className="w-4 h-4" />
              Buscar Huella
            </Button>
            <Button
              size="lg"
              onClick={() => navigate("/enroll")}
              className="gap-2"
            >
              <Fingerprint className="w-4 h-4" />
              Enrolar Huella
            </Button>
          </div>
        </header>

        {/* Content */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold">Casos Activos</h2>
            {caseList && (
              <span className="text-sm text-muted-foreground">
                {caseList.total} caso{caseList.total !== 1 ? "s" : ""}
              </span>
            )}
          </div>

          {isLoading && <LoadingSkeleton />}

          {isError && (
            <Card className="border-destructive/50 bg-destructive/5">
              <CardContent className="flex items-center gap-3 py-6">
                <AlertCircle className="w-5 h-5 text-destructive shrink-0" />
                <div>
                  <p className="font-medium text-destructive">
                    Error al cargar casos
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {error instanceof Error
                      ? error.message
                      : "No se pudo conectar con el servidor"}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}

          {!isLoading && !isError && !caseList?.items.length && <EmptyState />}

          {!isLoading && !isError && caseList && caseList.items.length > 0 && (
            <div className="space-y-4">
              {caseList.items.map((caseItem) => (
                <CaseCard key={caseItem.id} caseItem={caseItem} />
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
